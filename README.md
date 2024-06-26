---
title: Vista
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability

https://github.com/rerun-io/hf-example-vista/assets/9785832/0b9a01ca-90a2-4b36-98fc-a7a7b378fd54

[Shenyuan Gao](https://github.com/Little-Podi), [Jiazhi Yang](https://scholar.google.com/citations?user=Ju7nGX8AAAAJ&hl=en), [Li Chen](https://scholar.google.com/citations?user=ulZxvY0AAAAJ&hl=en), [Kashyap Chitta](https://kashyap7x.github.io/), [Yihang Qiu](https://scholar.google.com/citations?user=qgRUOdIAAAAJ&hl=en), [Andreas Geiger](https://www.cvlibs.net/), [Jun Zhang](https://eejzhang.people.ust.hk/), [Hongyang Li](https://lihongyang.info/)

This is a demo of the [Vista model](https://github.com/OpenDriveLab/Vista), a driving world model that can be used to simulate a variety of driving scenarios. This demo uses [Rerun](https://rerun.io/)'s custom [gradio component](https://www.gradio.app/custom-components/gallery?id=radames%2Fgradio_rerun) to livestream the model's output and show intermediate results.

[📜technical report](https://arxiv.org/abs/2405.17398),  [🎬video demos](https://vista-demo.github.io/),  [🤗model weights](https://huggingface.co/OpenDriveLab/Vista)

Please refer to the [original repository](https://github.com/OpenDriveLab/Vista) for the original code base and README.

You can try the example on Rerun's HuggingFace space [here](https://huggingface.co/spaces/rerun/Vista).

## Run the example locally
To run this example locally use the following command (you need a GPU with at least 20GB of memory, tested with an RTX 4090):
```bash
pixi run example
```

You can specify the first image, the number of generated segments, and the number of diffusion steps per segment:
```bash
pixi run example --img-path "example_images/streetview.png" --num-segments 10 --num-steps 100
```

To see other all options, use the following:
```bash
pixi run example --help
```
