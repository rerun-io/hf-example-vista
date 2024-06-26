"""Gradio interface for Vista model."""
from __future__ import annotations

import glob
import os
import queue
import threading

import gradio as gr
import gradio_rerun
import rerun as rr
import spaces

import vista


@spaces.GPU(duration=400)
@rr.thread_local_stream("Vista")
def generate_gradio(
    first_frame_file_name: str,
    n_rounds: float=3,
    n_steps: float=10,
    height=576,
    width=1024,
    n_frames=25,
    cfg_scale=2.5,
    cond_aug=0.0,
):
    global model

    n_rounds = int(n_rounds)
    n_steps = int(n_steps)

    # Use a queue to log immediately from internals
    log_queue = queue.SimpleQueue()

    stream = rr.binary_stream()

    blueprint = vista.generate_blueprint(n_rounds)
    rr.send_blueprint(blueprint)
    yield stream.read()

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
            model,
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
            yield stream.read()
    handle.join()


model = vista.create_model()

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(
        """
        # Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability

        [Shenyuan Gao](https://github.com/Little-Podi), [Jiazhi Yang](https://scholar.google.com/citations?user=Ju7nGX8AAAAJ&hl=en), [Li Chen](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en), [Kashyap Chitta](https://kashyap7x.github.io/), [Yihang Qiu](https://scholar.google.com/citations?user=qgRUOdIAAAAJ&hl=en), [Andreas Geiger](https://www.cvlibs.net/), [Jun Zhang](https://eejzhang.people.ust.hk/), [Hongyang Li](https://lihongyang.info/)

        This is a demo of the [Vista model](https://github.com/OpenDriveLab/Vista), a driving world model that can be used to simulate a variety of driving scenarios. This demo uses [Rerun](https://rerun.io/)'s custom [gradio component](https://www.gradio.app/custom-components/gallery?id=radames%2Fgradio_rerun) to livestream the model's output and show intermediate results.

         [📜technical report](https://arxiv.org/abs/2405.17398),  [🎬video demos](https://vista-demo.github.io/),  [🤗model weights](https://huggingface.co/OpenDriveLab/Vista)

         Note that the GPU time is limited to 400 seconds per run. If you need more time, you can run the model locally or on your own server.
        """
    )
    first_frame = gr.Image(sources="upload", type="filepath")
    example_dir_path = os.path.join(os.path.dirname(__file__), "example_images")
    example_file_paths = sorted(glob.glob(os.path.join(example_dir_path, "*.*")))
    example_gallery = gr.Examples(
        examples=example_file_paths,
        inputs=first_frame,
        cache_examples=False,
    )

    btn = gr.Button("Generate video")
    num_rounds = gr.Slider(
        label="Segments",
        info="Number of 25 frame segments to generate. Higher values lead to longer videos. Try to keep the product of segments and steps below 30 to avoid running out of time.",
        minimum=1,
        maximum=5,
        value=2,
        step=1
    )
    num_steps = gr.Slider(
        label="Diffusion Steps",
        info="Number of diffusion steps per segment. Higher values lead to more detailed videos. Try to keep the product of segments and steps below 30 to avoid running out of time.",
        minimum=1,
        maximum=50,
        value=15,
        step=1
    )

    with gr.Row():
        viewer = gradio_rerun.Rerun(streaming=True)
    btn.click(
        generate_gradio,
        inputs=[first_frame, num_rounds, num_steps],
        outputs=[viewer],
    )

demo.launch()
