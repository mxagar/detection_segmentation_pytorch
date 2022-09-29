# Beyond Image Classification: Object Detection and Semantic Segmentation with Pytorch

This repository collects examples and resources in which [*Object Detection*](https://en.wikipedia.org/wiki/Object_detection) and [*Semantic Segmentation*](https://en.wikipedia.org/wiki/Image_segmentation) are applied to images with [Pytorch](https://pytorch.org/).

:warning: **Important** :warning:

- This is an on-going project; I will extend the content as far as I have time for it.
- In some cases, I will use the code from other public tutorials/posts rightfully citing the source.
- In addition to the examples, the links in the section [Interesting Links](#interesting-links) are very useful resources for those interested in the topic.

Probably the most known application of Convolutional Neural Networks is *Image Classification*: an image is convoluted with several trained filters to output a vector of `N` values; each value is a float between `0` and `1` and it denotes the probability of the image containing one of the pre-defined `N` classes of objects.

However, there are also CNN architectures and methods that target other applications:

- Object Detection: given an image and a predefined set of `N` classes, we obtain the bounding boxes (typically `[x,y,w,h]` vectors) that enclose known object classes, and for each bounding box, a vector of length `N` which determines the probabilities of each class, as in image classification.
- Semantic Segmentation: given an image and a predefined set of `N` classes, we obtain the `N` class probabilities for each pixel; i.e., we can segment regions in the images.

The following figure illustrates the difference between the three techniques:

![Deep Learning on Images: Methods](./assets/Deep_Learning_on_Images_Methods.png)

As mentioned, this repository collects practical examples that target the last two applications.

Table of contents:
- [Beyond Image Classification: Object Detection and Semantic Segmentation with Pytorch](#beyond-image-classification-object-detection-and-semantic-segmentation-with-pytorch)
  - [Overview and File Structure](#overview-and-file-structure)
    - [How to Use This](#how-to-use-this)
    - [Dependencies](#dependencies)
  - [Object Detection: General Notes](#object-detection-general-notes)
  - [Semantic Segmentation: General Notes](#semantic-segmentation-general-notes)
  - [List of Examples + Description Points](#list-of-examples--description-points)
  - [Improvements and Possible Extensions](#improvements-and-possible-extensions)
  - [Interesting Links](#interesting-links)
  - [Authorship](#authorship)

## Overview and File Structure

### How to Use This

TBD.

### Dependencies

You should create a python environment (e.g., with [conda](https://docs.conda.io/en/latest/)) and install the dependencies listed in the [requirements.txt](requirements.txt) file of each example. If there is no such file in a folder example, the one in the root level should work.

A short summary of commands required to have all in place is the following:

```bash
conda create -n det-seg python=3.6
conda activate det-seg
conda install pytorch torchvision -c pytorch 
conda install pip
pip install -r requirements.txt
```

## Object Detection: General Notes

TBD.
## Semantic Segmentation: General Notes

TBD.

## List of Examples + Description Points

- [`01_mask_r_cnn_fine_tuning`](01_mask_r_cnn_fine_tuning)
- [`02_yolo_v3_darknet`](02_yolo_v3_darknet)

## Improvements and Possible Extensions

- [`01_mask_r_cnn_fine_tuning`](01_mask_r_cnn_fine_tuning)
  - [ ] TBD.
- [`02_yolo_v3_darknet`](02_yolo_v3_darknet)
  - [ ] TBD.

## Interesting Links

Some **sources** I have used:

- [TorchVision Object Detection Finetuning - Pytorch Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [Segmentation Models: Python Library](https://github.com/qubvel/segmentation_models.pytorch)
- [PyImageSearch: U-Net: Training Image Segmentation Models in PyTorch](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)
- [Image Segmentation DeepLabV3 on iOS - Pytorch Tutorial](https://pytorch.org/tutorials/beginner/deeplabv3_on_ios.html)

**My related notes**:

- [My notes and code](https://github.com/mxagar/computer_vision_udacity) on the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).
- [My notes and code](https://github.com/mxagar/deep_learning_udacity) on the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

Other **resources and tutorials**:

- [Deep Learning for Object Detection: A Comprehensive Review](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)
- [A Faster Pytorch Implementation of Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch)
- [Train your own object detector with Faster-RCNN & PyTorch](https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70)
- [Creating and training a U-Net model with PyTorch for 2D & 3D semantic segmentation: Dataset building](https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55)

**Papers**: look in the folder [literature](literature/README.md)

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to use this project, but please link it back to the original source.