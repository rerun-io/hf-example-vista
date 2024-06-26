from __future__ import annotations

import rerun.blueprint as rrb
import torch
from transformers.utils import hub

from . import sample, sample_utils


def create_model():
    return sample_utils.init_model(
        {
            "config": "./vista/configs/inference/vista.yaml",
            "ckpt": hub.get_file_from_repo("OpenDriveLab/Vista", "vista.safetensors"),
        }
    )


def generate_blueprint(n_rounds: int) -> rrb.Blueprint:
    row1 = rrb.Horizontal(
        *[
            rrb.TensorView(origin=f"diffusion_{i}", name=f"Latents Segment {i+1}")
            for i in range(n_rounds)
        ],
    )
    row2 = rrb.Spatial2DView(origin="generated_image", name="Generated Video")

    return rrb.Blueprint(rrb.Vertical(row1, row2), collapse_panels=True)


def run_sampling(
    log_queue,
    first_frame_file_name,
    height,
    width,
    n_rounds,
    n_frames,
    n_steps,
    cfg_scale,
    cond_aug,
    model=None,
) -> None:
    if model is None:
        model = create_model()

    unique_keys = set([x.input_key for x in model.conditioner.embedders])
    value_dict = sample_utils.init_embedder_options(unique_keys)

    action_dict = None

    first_frame = sample.load_img(first_frame_file_name, height, width, "cuda")[None]
    repeated_frame = first_frame.expand(n_frames, -1, -1, -1)

    value_dict = sample_utils.init_embedder_options(unique_keys)
    cond_img = first_frame
    value_dict["cond_frames_without_noise"] = cond_img
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames"] = cond_img + cond_aug * torch.randn_like(cond_img)
    if action_dict is not None:
        for key, value in action_dict.items():
            value_dict[key] = value

    if n_rounds > 1:
        guider = "TrianglePredictionGuider"
    else:
        guider = "VanillaCFG"
    sampler = sample_utils.init_sampling(
        guider=guider,
        steps=n_steps,
        cfg_scale=cfg_scale,
        num_frames=n_frames,
    )

    uc_keys = [
        "cond_frames",
        "cond_frames_without_noise",
        "command",
        "trajectory",
        "speed",
        "angle",
        "goal",
    ]

    _generated_images, _samples_z, _inputs = sample_utils.do_sample(
        repeated_frame,
        model,
        sampler,
        value_dict,
        num_rounds=n_rounds,
        num_frames=n_frames,
        force_uc_zero_embeddings=uc_keys,
        initial_cond_indices=[0],
        log_queue=log_queue,
    )

    log_queue.put("done")
