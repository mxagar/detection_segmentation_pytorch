# Mask R-CNN for Person Detection and Segmentation

This tutorial has been downloaded from a [Pytorch tutorial post](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), which contains a link to the [Colab version](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb) of the example.

In it, a pre-trained [Mask R-CNN](https://arxiv.org/abs/1703.06870) model is fine-tuned with the [*Penn-Fudan Database for Pedestrian Detection and Segmentation*](https://www.cis.upenn.edu/~jshi/ped_html/). It contains 170 images with 345 instances of pedestrians.

The only required file is the notebook [torchvision_finetuning_instance_segmentation.ipynb](torchvision_finetuning_instance_segmentation.ipynb); this notebook:

- Installs additional packages: `cython` and `cocoapi`.
- Downloads the dataset, required util scripts, etc.
  - `PennFudanPed`
  - `coco_*.py`, `engine.py`, `transforms.py`, etc.
- Explains what's done, as in the post.

Additionally, the script [tv-training-code.py](tv-training-code.py) contains the key functions from the notebook that carry out the main tasks.

:warning: **Important Notes**

- I had issues running the notebook on my Macbook Pro M1 and on my Jetson Nano, so I replicated the environment in Google Colab. However, note that the training is extremely slow on machines without a proper CUDA; thus, the pushed version is probably a notebook executed in Google Colab.
- I had to change the `engine.py` file: the decorator `@torch.no_grad()` is commented out and I use a context manager outside for `evaluate()`; additionally, I check `torch.cuda.is_available()` before anything related to CUDA.

### Installation and Usage

```bash
conda create -n mask-rcnn python=3.7.14
conda activate mask-rcnn
conda install pytorch torchvision -c pytorch 
conda install pip
pip install -r requirements.txt
jupyter notebook torchvision_finetuning_instance_segmentation.ipynb
```

... or, alternatively:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mxagar/detection_segmentation_pytorch/blob/main/01_mask_r_cnn_fine_tuning/torchvision_finetuning_instance_segmentation.ipynb)

