# SAM: Segment Anything Model from Facebook/Meta Research

[Segment Anything Model](https://github.com/facebookresearch/segment-anything)

[SAM Paper](https://scontent-bcn1-1.xx.fbcdn.net/v/t39.2365-6/10000000_900554171201033_1602411987825904100_n.pdf?_nc_cat=100&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=qTvYBYD2upoAX8voNfw&_nc_ht=scontent-bcn1-1.xx&oh=00_AfD_xIHUhJt9Lm70nBOP4mrLHRfEiJyb8v0uFYehP4IDlw&oe=6493ECE7)

## 1. Introduction and Setup

Even though I have my tests here, the model package and the model weights are located in a folder which is the result of cloning the original model repository:

`~/git_repositories/segment-anything`

In the following, the overall setup is explained.

Basic environment setup:

```bash
# Crate env: requirements in conda.yaml
# This packages are the basic for Pytorch-CUDA usage
# HOWEVER, we need to install more later on
# when we clone and use the original SAM repo
conda env create -f conda.yaml
conda activate sam

# Pytorch: Windows + CUDA 11.7
# Update your NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
# I have version 12.1, but it works with older versions, e.g. 11.7
# Check your CUDA version with: nvidia-smi.exe
# In case of any runtime errors, check vrsion compatibility tables:
# https://github.com/pytorch/vision#installation
# The default conda installation command DID NOT WORK
conda install pytorch=1.12 torchvision=0.13 torchtext=0.13 torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
# But the following pip install command DID WORK
python -m pip install torch==1.13+cu117 torchvision==0.14+cu117 torchaudio torchtext==0.14 --index-url https://download.pytorch.org/whl/cu117

# Pytorch: Mac / Windows CPU
python -m pip install torch torchvision torchaudio

# Dump installed libraries in pip format
python -m pip list --format=freeze > requirements.txt
```

Additionally, I [cloned and installed the model repository](https://github.com/facebookresearch/segment-anything#installation):

```bash
cd ~/git_repositories

git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything

conda activate sam
pip install -e .

pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

Finally, we need to download the [model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints), located in the non-committed folder `model-checkpoints`.

## 2. Tests

The folder [`notebooks`](./notebooks/) contains the official example notebooks from the Facebook/Meta repository, which I have tried in this space:

- [`predictor_example.ipynb`](./notebooks/predictor_example.ipynb)
- [`automatic_mask_generator_example.ipynb`](./notebooks/automatic_mask_generator_example.ipynb)
- [`onnx_model_example.ipynb`](./notebooks/onnx_model_example.ipynb)

### Predictor Example

TBD.

[`predictor_example.ipynb`](./notebooks/predictor_example.ipynb)

:construction:

### Automatic Mask Generator Example

TBD.

[`automatic_mask_generator_example.ipynb`](./notebooks/automatic_mask_generator_example.ipynb)

:construction:

### ONNX Model Example

TBD.

[`onnx_model_example.ipynb`](./notebooks/onnx_model_example.ipynb)

:construction:
