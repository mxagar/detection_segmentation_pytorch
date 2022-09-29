# Mask R-CNN for Person Detection and Segmentation

This tutorial has been downloaded from a [Pytorch tutorial post](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), which contains a link to the [Colab version](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb) of the example.

In it, a pre-trained [Mask R-CNN](https://arxiv.org/abs/1703.06870) model is fine-tuned with the [*Penn-Fudan Database for Pedestrian Detection and Segmentation*](https://www.cis.upenn.edu/~jshi/ped_html/). It contains 170 images with 345 instances of pedestrians.

The only required file is the notebook [torchvision_finetuning_instance_segmentation.ipynb](torchvision_finetuning_instance_segmentation.ipynb); this notebook:

- Installs additional packages.
- Downloads the dataset, required util scripts, etc.
- Explains what's done, as in the post.