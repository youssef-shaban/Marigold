# Marigold Computer Vision

This repository provides a focused Marigold setup for surface normals estimation on the AIM2PC dataset. It keeps the
normals pipeline and training/inference scripts, while depth and intrinsic-image components are intentionally removed.

## Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis

[![Website](doc/badges/badge-website.svg)](https://marigoldcomputervision.github.io)
[![Paper](doc/badges/badge-pdf.svg)](https://arxiv.org/abs/2505.09358)
[![Normals Demo](https://img.shields.io/badge/🤗%20Normals-Demo-yellow)](https://huggingface.co/spaces/prs-eth/marigold-normals)
[![Normals Model](https://img.shields.io/badge/🤗%20Normals-Model-green)](https://huggingface.co/prs-eth/marigold-normals-v1-1)
[![Diffusers Tutorial](doc/badges/badge-hfdiffusers.svg)](https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage)

Team:
[Bingxin Ke](http://www.kebingxin.com/),
[Kevin Qu](https://www.linkedin.com/in/kevin-qu-b3417621b/),
[Tianfu Wang](https://tianfwang.github.io/)
[Nando Metzger](https://nandometzger.github.io/),
[Shengyu Huang](https://shengyuh.github.io/),
[Bo Li](https://www.linkedin.com/in/bobboli0202/),
[Anton Obukhov](https://www.obukhov.ai/),
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ)

We present Marigold, a family of conditional generative models and a fine-tuning protocol that adapts latent diffusion
models for dense image analysis tasks. This fork focuses on surface normal prediction and aims to keep the codebase
minimal for that task.

![teaser_all](doc/teaser_marigold_all.jpg)

## 📢 News
2025-05-15: Released code and a [checkpoint](https://huggingface.co/prs-eth/marigold-normals-v1-1) of Marigold Surface Normals Estimation (v1.1).<br>
2024-05-28: Training code is released.<br>
2024-05-27: Marigold pipelines are merged into the `diffusers` core starting v0.28.0 [release](https://github.com/huggingface/diffusers/releases/tag/v0.28.0)!<br>
2024-03-04: The paper is accepted at CVPR 2024.<br>
2023-12-19: Updated [license](LICENSE.txt) to Apache License, Version 2.0.<br>
2023-12-08: Added the first interactive [Hugging Face Space Demo](https://huggingface.co/spaces/prs-eth/marigold) of depth estimation.<br>
2023-12-05: Added a [Google Colab](https://colab.research.google.com/drive/12G8reD13DdpMie5ZQlaFNo2WCGeNUH-u?usp=sharing)<br>
2023-12-04: Added an [arXiv paper](https://arxiv.org/abs/2312.02145) and inference code (this repository).

## 🚀 Usage

**We offer several ways to interact with Marigold**:

1. A free online interactive demo:
<a href="https://huggingface.co/spaces/prs-eth/marigold-normals"><img src="https://img.shields.io/badge/🤗%20Normals-Demo-yellow" height="16"></a>

1. Marigold pipelines are part of
<a href="https://huggingface.co/docs/diffusers/using-diffusers/marigold_usage"><img src="doc/badges/badge-hfdiffusers.svg" height="16"></a> - a one-stop shop for diffusion 🧨!

1. Run the demo locally (requires a GPU and an `nvidia-docker2`, see [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):
`docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all registry.hf.space/prs-eth-marigold:latest python app.py`

1. Extended demo on a Google Colab: <a href="https://colab.research.google.com/drive/12G8reD13DdpMie5ZQlaFNo2WCGeNUH-u?usp=sharing"><img src="doc/badges/badge-colab.svg" height="16"></a>

1. If you just want to see the examples, visit our gallery: <a href="https://marigoldcomputervision.github.io"><img src="doc/badges/badge-website.svg" height="16"></a>

1. Finally, local development instructions with this codebase are given below.

## 🛠️ Setup

The inference code was tested on:

- Ubuntu 22.04 LTS, Python 3.10.12,  CUDA 11.7, GeForce RTX 3090 (pip)

### 🪧 A Note for Windows users

We recommend running the code in WSL2:

1. Install WSL following [installation guide](https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command).
1. Install CUDA support for WSL following [installation guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2).
1. Find your drives in `/mnt/<drive letter>/`; check [WSL FAQ](https://learn.microsoft.com/en-us/windows/wsl/faq#how-do-i-access-my-c--drive-) for more details. Navigate to the working directory of choice. 

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/prs-eth/Marigold.git
cd Marigold
```

### 💻 Dependencies

Install the dependencies:

```bash
python -m venv venv/marigold
source venv/marigold/bin/activate
pip install -r requirements.txt
```

Keep the environment activated before running the inference script. 
Activate the environment again after restarting the terminal session.

## 🏃 Testing on your images

### 📷 Prepare images

Use selected images from our paper:

```bash
bash script/download_sample_data.sh
```

Or place your images in a directory, for example, under `input/in-the-wild_example`, and run the following inference command.

### 🚀 Run inference (for practical usage)

```bash
# Normals
python script/normals/run.py \
    --checkpoint prs-eth/marigold-normals-v1-1 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example \
    --fp16
```

### ⚙️ Inference settings

The default settings are optimized for the best results. However, the behavior of the code can be customized:

- `--half_precision` or `--fp16`: Run with half-precision (16-bit float) to reduce VRAM usage (may reduce quality).

- `--ensemble_size`: Number of inference passes in the ensemble. Larger values give better results at a cost; default is 1.

- `--denoise_steps`: Number of denoising diffusion steps. Defaults are stored in the checkpoint.

- By default, the inference script resizes input images to the *processing resolution*, and then resizes the prediction back to the original resolution.  
  
  - `--processing_res`: processing resolution; set to 0 to process the input resolution directly. When `None`, reads the default from the model config.
  - `--output_processing_res`: output at the processing resolution instead of upsampling to input resolution.
  - `--resample_method`: resampling method used to resize images and normals predictions. One of `bilinear`, `bicubic`, `nearest`.

- `--seed`: Random seed for reproducibility. Default: None (unseeded). For full reproducibility, see [deterministic mode](https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms).
- `--batch_size`: Batch size of repeated inference. Default: 0 (auto).
- `--apple_silicon`: Use Apple Silicon MPS acceleration.
- `--use_aspect_ratio`: Enable aspect-ratio conditioning when supported by the model.


### 🎮 Run inference (for academic comparisons)

These settings correspond to our paper. For academic comparison, please run with the settings below (if you only want to do fast inference on your own images, you can set `--ensemble_size 1`).

```bash
# Normals
python script/normals/run.py \
    --checkpoint prs-eth/marigold-normals-v1-1 \
    --denoise_steps 4 \
    --ensemble_size 10 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example
```

You can find all results in the `output` directory. Enjoy!


### ⬇ Checkpoint cache

By default, the normals checkpoint ([marigold-normals-v1-1](https://huggingface.co/prs-eth/marigold-normals-v1-1)) is stored in the Hugging Face cache.
The `HF_HOME` environment variable defines its location and can be overridden, e.g.:

```bash
export HF_HOME=$(pwd)/cache
```

Alternatively, use the following script to download the checkpoint weights locally:

```bash
bash script/download_weights.sh marigold-normals-v1-1
```

At inference, specify the checkpoint path:

```bash
# Normals
python script/normals/run.py \
    --checkpoint checkpoint/marigold-normals-v1-1 \
    --denoise_steps 4 \
    --ensemble_size 1 \
    --input_rgb_dir input/in-the-wild_example \
    --output_dir output/in-the-wild_example
```

## 🦿 Evaluation on test datasets <a name="evaluation"></a>
Install additional dependencies:

```bash
pip install -r requirements+.txt -r requirements.txt
``` 

Set data directory variable (also needed in evaluation scripts) and point it to your AIM2PC dataset:

```bash
export BASE_DATA_DIR=<YOUR_DATA_DIR>  # Set target data directory
```
Run inference and evaluation scripts, for example:

```bash
# Normals (AIM2PC config)
python script/normals/infer.py \
    --checkpoint prs-eth/marigold-normals-v1-1 \
    --dataset_config config/dataset_normals/dataset_val.yaml \
    --base_data_dir ${BASE_DATA_DIR} \
    --output_dir output/aim2pc_eval \
    --denoise_steps 4 \
    --processing_res 512 \
    --ensemble_size 1

python script/normals/eval.py \
    --prediction_dir output/aim2pc_eval/normals_latent \
    --dataset_config config/dataset_normals/dataset_val.yaml \
    --base_data_dir ${BASE_DATA_DIR} \
    --output_dir output/aim2pc_eval_metrics
```

Note: although the seed has been set, the results might still be slightly different on different hardware.

## 🏋️ Training

Based on the previously created environment, install extended requirements:

```bash
pip install -r requirements++.txt -r requirements+.txt -r requirements.txt
```

Set environment parameters for the data directory:

```bash
export BASE_DATA_DIR=YOUR_DATA_DIR        # directory of training data
export BASE_CKPT_DIR=YOUR_CHECKPOINT_DIR  # directory of pretrained checkpoint
```

Download Stable Diffusion v2 [checkpoint](https://huggingface.co/stabilityai/stable-diffusion-2) into `${BASE_CKPT_DIR}`

### Prepare for training data
Prepare the AIM2PC dataset with the following structure under `${BASE_DATA_DIR}`:

```
aim2pc_normals/
  train/
  val/
  vis/
```

Each split should contain `image/`, `mesh/`, and `roof_intuitive_mask/` subfolders as expected by `AIM2PCNormalsDataset`.


### Run training script

```bash
# Normals
python script/normals/train.py --config config/train_marigold_normals.yaml
```

Resume from a checkpoint, e.g.:

```bash
# Normals
python script/normals/train.py --resume_run output/train_marigold_normals/checkpoint/latest
```

### Compose checkpoint:
Only the U-Net and scheduler config are updated during training. They are saved in the training directory. To use the inference pipeline with your training result:
- replace `unet` folder in Marigold checkpoints with that in the `checkpoint` output folder.
- replace the `scheduler/scheduler_config.json` file in Marigold checkpoints with `checkpoint/scheduler_config.json` generated during training.
Then refer to [this section](#evaluation) for evaluation.

**Note**: Although random seeds have been set, the training result might be slightly different on different hardwares. It's recommended to train without interruption.

## ✏️ Contributing

Please refer to [this](CONTRIBUTING.md) instruction.

## 🤔 Troubleshooting

| Problem                                                                                                                                      | Solution                                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| (Windows) Invalid DOS bash script on WSL                                                                                                     | Run `dos2unix <script_name>` to convert script format          |
| (Windows) error on WSL: `Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory` | Run `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` |
| Training takes a long time to start | Use folders for data instead of tar files (modification in config files is required).  |



## 🎓 Citation

Please cite our papers:

```bibtex
@InProceedings{ke2023repurposing,
  title={Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation},
  author={Bingxin Ke and Anton Obukhov and Shengyu Huang and Nando Metzger and Rodrigo Caye Daudt and Konrad Schindler},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}

@misc{ke2025marigold,
  title={Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis},
  author={Bingxin Ke and Kevin Qu and Tianfu Wang and Nando Metzger and Shengyu Huang and Bo Li and Anton Obukhov and Konrad Schindler},
  year={2025},
  eprint={2505.09358},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## 🎫 License

This code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

The models are licensed under RAIL++-M License (as defined in the [LICENSE-MODEL](LICENSE-MODEL.txt))

By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE.txt) and [LICENSE-MODEL](LICENSE-MODEL.txt) respectively.
