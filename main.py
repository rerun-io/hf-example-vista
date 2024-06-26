"""Command line interface for generating videos from the model."""
from __future__ import annotations

import argparse
import queue
import threading

import rerun as rr

import vista


def generate_local(
    first_frame_file_name: str,
    height=576,
    width=1024,
    n_rounds=4,
    n_frames=25,
    n_steps=10,
    cfg_scale=2.5,
    cond_aug=0.0,
):
    # Use a queue to log immediately from internals
    log_queue = queue.SimpleQueue()

    handle = threading.Thread(
        target=vista.run_sampling,
        args=[
            log_queue,
            first_frame_file_name,
            height,
            width,
            n_rounds,
            n_frames,
            n_steps,
            cfg_scale,
            cond_aug,
        ],
    )
    handle.start()
    while True:
        msg = log_queue.get()
        if msg == "done":
            break
        else:
            entity_path, entity, times = msg
            rr.reset_time()
            for timeline, time in times:
                if isinstance(time, int):
                    rr.set_time_sequence(timeline, time)
                else:
                    rr.set_time_seconds(timeline, time)
            rr.log(entity_path, entity)
    handle.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate video conditioned on a single image using the Vista model."
    )
    parser.add_argument(
        "--img-path",
        type=str,
        help="Path to image used as input for Canny edge detector.",
        default="./example_images/nus-0.jpg",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Number of diffusion steps per image. Recommended range: 10-50. Higher values result in more detailed images and less blurry results.",
        default=20,
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        help="Number of segments to generate. Each segment consists of 25 frames.",
        default=3,
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(
        args,
        "rerun_example_vista",
        default_blueprint=vista.generate_blueprint(args.num_segments),
    )

    generate_local(args.img_path, n_steps=args.num_steps, n_rounds=args.num_segments)
