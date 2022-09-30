# Beyond Image Classification: Object Detection and Semantic Segmentation with Pytorch

This repository collects examples and resources in which [*Object Detection*](https://en.wikipedia.org/wiki/Object_detection) and [*Semantic Segmentation*](https://en.wikipedia.org/wiki/Image_segmentation) are applied to images with [Pytorch](https://pytorch.org/).

:warning: Important notes, first of all:

[![Unfinished](https://img.shields.io/badge/status-unfinished-orange)](https://shields.io/#your-badge)

- This is an on-going project; I will extend the content as far as I have time for it.
- In some cases, I will use the code from other public tutorials/posts rightfully citing the source.
- In addition to the examples, the links in the section [Interesting Links](#interesting-links) are very useful resources for those interested in the topic.

## Introduction

Probably the most known application of Convolutional Neural Networks is *Image Classification*: an image is convoluted with several trained filters to output a vector of `N` values; each value is a float between `0` and `1` and it denotes the probability of the image containing one of the pre-defined `N` classes of objects.

However, there are also CNN architectures and methods that target other applications:

- Object Detection: given an image and a predefined set of `N` classes, we obtain the bounding boxes (typically `[x,y,w,h]` vectors) that enclose known object classes, and for each bounding box, a vector of length `N` which determines the probabilities of each class, as in image classification.
- Semantic Segmentation: given an image and a predefined set of `N` classes, we obtain the `N` class probabilities for each pixel; i.e., we can segment regions in the images.

The following figure illustrates the difference between the three techniques:

![Deep Learning on Images: Methods](./assets/Deep_Learning_on_Images_Methods.png)

As mentioned, this repository collects practical examples that target the last two applications.

### Table of Contents
- [Beyond Image Classification: Object Detection and Semantic Segmentation with Pytorch](#beyond-image-classification-object-detection-and-semantic-segmentation-with-pytorch)
  - [Introduction](#introduction)
    - [Table of Contents](#table-of-contents)
  - [Overview and File Structure](#overview-and-file-structure)
    - [How to Use This](#how-to-use-this)
    - [Dependencies](#dependencies)
  - [Object Detection: General Notes](#object-detection-general-notes)
    - [Faster R-CNN](#faster-r-cnn)
    - [YOLO: You Only Look Once](#yolo-you-only-look-once)
  - [Semantic Segmentation: General Notes](#semantic-segmentation-general-notes)
  - [List of Examples + Description Points](#list-of-examples--description-points)
  - [Improvements and Possible Extensions](#improvements-and-possible-extensions)
  - [Interesting Links](#interesting-links)
  - [Authorship](#authorship)

## Overview and File Structure

### How to Use This

1. Go to the desired example folder from the section [List of Examples + Description Points](#list-of-examples--description-points). You should have brief instructions in each folder.
2. If there is an `Open in Colab` button anywhere, you can use it :smile:.
3. If you'd like to run the code locally on your device install the [dependencies](#dependencies) and run the main file in it; often, the main file will be a notebook that takes care of all.

### Dependencies

You should create a python environment (e.g., with [conda](https://docs.conda.io/en/latest/)) and install the dependencies listed in the [requirements.txt](requirements.txt) file of each example. If there is no such file in a folder example, the one in the root level should work.

A short summary of commands required to have all in place is the following; however, as mentioned, **each example might have its own specific dependency versions**:

```bash
conda create -n det-seg python=3.6
conda activate det-seg
conda install pytorch torchvision -c pytorch 
conda install pip
pip install -r requirements.txt
```

## Object Detection: General Notes

In this section, I provide some high level notes on the theory behind the object detection networks; for more details, check the articles listed in the [literature](literature) folder.

Object detection networks can use classification backbones to extract features, but at the end of the feature extractor, instead of mapping the activations to classes, they perform a more sophisticated tasks, with which they basically:

- they regress Regions of Interest (ROI), i.e., bounding boxes, that likely contain an object
- and they yield class probabilities for those bounding boxes.

### Faster R-CNN

One type of architecture for object detection is **Region-CNN**, or **R-CNN**; there are several versions of it, but the one typically used (due to its performance) is the **Faster R-CNN**.

The **Faster R-CNN** network has the following steps:

- The image is convoluted until a given layer, which produces a set of feature maps.
- We feed the feature maps to a separate network which predicts possible ROIs: these are called **Region Proposal Networks**; if edges or other relevant features have been detected, ROIs that enclose them will emerge.
- The ROI proposals are passed to the original network, which performs a quick binary check: does the ROI contain an object or not? If so, the ROI is taken.
- For each ROI, ROI pooling is performed: non-uniform cells of pooling are applied to warp the ROIs to standard sizes
- The last part of the network predicts the class of the ROI.

The main difference between the R-CNN architecture lies on the ROI detection: less efficient networks project ROIs computed after applying classical algorithms, while the Faster R-CNN uses a Region Proposal Network.

Region proposal networks work as follows:

- A small (usually 3x3) window is slided on the feature maps.
- `k` anchor boxes are applied on each window. These anchor boxes are pre-defined boxes with different aspect ratios.
- For each `k` boxes in each window, the probability of it containing an object is measured. If it's higher than a threshold, the anchor box is suggested as a ROI.

During training, the ground truth is given by the real bounding box: if the suggested ROI overlaps considerably with a true bounding box, the suggestion is correct.




### YOLO: You Only Look Once


## Semantic Segmentation: General Notes

In this section, I provide some high level notes on the theory behind the semantic segmentation networks; for more details, check the articles listed in the [literature](literature) folder.


TBD.

## List of Examples + Description Points

- [`01_mask_r_cnn_fine_tuning`](01_mask_r_cnn_fine_tuning)
  - (Object) detection and segmentation of humans.
  - A custom data loader is defined for segmentation and detection.
  - An hybrid model is created which predicts the bounding box (Faster-RCNN) and segments the object in it (Mask-RCNN).
  - We have also examples about how to:
    - Use only Faster-RCNN for our custom classes
    - Use another backbone for faster inferences, e.g., MobileNet insteead of the default ResNet50.
  - Helper scripts are downloaded, which facilitate the training.
  - Training is performed and the model evaluated.

- [`02_yolo_v3_darknet`](02_yolo_v3_darknet)
  - Object detection.
  - A custom wrapper for YOLO-v3 is provided: a Pytorch model is created, darknet weights read and loaded into the model.
  - Only inference is possible, not training.
  - The COCO classes can be used only.

## Improvements and Possible Extensions

- [`01_mask_r_cnn_fine_tuning`](01_mask_r_cnn_fine_tuning)
  - [ ] Isolate the detection model and use `FastRCNNPredictor` to fine-tune my own classes. This point is suggested in the notebook.
- [`02_yolo_v3_darknet`](02_yolo_v3_darknet)
  - [ ] Get a YOLO model in Pytorch that can be trained.
  - [ ] Get a YOLO model which can be modified and trained for custom object classes.

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
- A Github repo with a peer reviewed implementation of the Faster R-CNN: [A Faster Pytorch Implementation of Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch)
- [Train your own object detector with Faster-RCNN & PyTorch](https://johschmidt42.medium.com/train-your-own-object-detector-with-faster-rcnn-pytorch-8d3c759cfc70)
- [Creating and training a U-Net model with PyTorch for 2D & 3D semantic segmentation: Dataset building](https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-dataset-fb1f7f80fe55)
- [A PyTorch Tutorial to Object Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection)

**Papers**: look in the folder [literature](literature/README.md).

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to use this project, but please link it back to the original source.