# Custom Object Detection with YOLO v7

I made these notes following these courses/tutorials by Nicolai Nielsen:

- [Nicolai Nielsen: YOLOv7 Custom Object Detection](https://nicolai-nielsen-s-school.teachable.com/courses)
- [Youtube: YOLO Object Detection Models, Nicolai Nielsen](https://www.youtube.com/playlist?list=PLkmvobsnE0GEfcliu9SXhtAQyyIiw9Kl0)

Table of contents:

- [Custom Object Detection with YOLO v7](#custom-object-detection-with-yolo-v7)
  - [1. Introduction](#1-introduction)
  - [2. YOLOv7 Architecture](#2-yolov7-architecture)
    - [Efficient Layer Aggregation](#efficient-layer-aggregation)
    - [Model Scaling Techniques](#model-scaling-techniques)
    - [Re-Parametrization Planning](#re-parametrization-planning)
    - [Auxiliary Head](#auxiliary-head)
  - [3. Custom Dataset](#3-custom-dataset)
    - [Capturing a Custom Dataset with OpenCV](#capturing-a-custom-dataset-with-opencv)
    - [Finding a Public Dataset](#finding-a-public-dataset)
    - [Labeling on Roboflow](#labeling-on-roboflow)
    - [Augmentation](#augmentation)
    - [Exporting the Dataset](#exporting-the-dataset)
  - [4. Training](#4-training)
  - [5. Deployment](#5-deployment)

## 1. Introduction

Requirements:

- [Roboflow](https://roboflow.com) account
- Anaconda
- Google Colab / Jupyter
- Python 3.9
- PyTorch (the newest version on Google Colab)
- OpenCV (all versions can be used, e.g., 4.5.2)
- Onnx runtime

Setup:

```bash
# Crate env: requirements in conda.yaml
conda env create -f conda.yaml
conda activate yolov7

# Pytorch: Windows + CUDA 11.7
# Update your NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
# I have version 12.1, but it works with older versions, e.g. 11.7
# Check your CUDA version with: nvidia-smi.exe
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Pytorch: Mac / Windows CPU
python -m pip install torch torchvision torchaudio

# Dump installed libraries in pip format
python -m pip list --format=freeze > requirements.txt
```

## 2. YOLOv7 Architecture

A concise explanation of **Object Detection** techniques is given in the [`README.md`](../README.md) from the upper level. Additionally, here some relevant links:

- [YOLOv7 Object Detection Paper Explanation and Inference](https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/)
- [Fine Tuning YOLOv7 on Custom Dataset](https://learnopencv.com/fine-tuning-yolov7-on-custom-dataset/)
- [Deep Learning for Object Detection: A Comprehensive Review](https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9)
- [Paper - YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

Main elements of the architecture:

- Backbone: feature extractor; we get feature maps of different sizes, all of a lower dimensionality.
- Feature Pyramid Network (FPN): we pass the feature maps to it; it has a pyramid-like structure, so features of different sizes are captured. The FPN seems to have routing / shortcuts.
- Heads: we have a head for each object/class allocation which predicts the object type and its bounding box.
- YOLO loss: each head has a YOLO loss with 3 components:
  - Cross-entropy, related to the class: `K` classes.
  - L1 loss, related to the bounding box: 4 values (`x, y, w, h`) 
  - Objectness loss: 1 value.

![YOLO v7 Architecture](./pics/yolo_v7_architecture.jpg)

In the following th emost important 

### Efficient Layer Aggregation

Extended efficient layer aggregation networks are used (E-ELAN), which increase the group cardinality.  "Cardinality" refers to the number of groups in group convolution. Group convolution is a technique where the input channels are divided into groups, and each group is convolved with a separate filter. By doing so, the computation cost of convolution can be reduced. The output feature maps of each group are then concatenated to form the final output.

The authors of the paper propose an extension to this technique, called extended efficient layer aggregation networks (E-ELAN), which uses group convolution to increase the cardinality of the added features. This means that they increase the number of groups in group convolution to learn more diverse features. They then combine the features of different groups in a "shuffle and merge cardinality manner," which means that they randomly shuffle the output feature maps of each group and then concatenate them together.

This approach is intended to enhance the features learned by different feature maps and improve the use of parameters and calculations, ultimately leading to better object detection performance.

![YOLO v7 Architecture: Efficient Layer Aggregation](./pics/yolo_v7_architecture_2.jpg)

### Model Scaling Techniques

Model scaling is achieved by increasing/decreasng both the depth and the width of the network layers, i.e., modifzing the number of channels, among others. These the effect of:

- Having models of different numbers of parameters, i.e., sizes
- Having models of different accuracies and speeds, i.e., they can be better suited for specific applications (e.g., realtime or offline but with higher accuracy).

![YOLO v7 Architecture: Scaling](./pics/yolo_v7_architecture_3.jpg)

YOLO uses concatenation-base models. These, also known as multi-scale feature fusion models, are neural networks that combine features extracted at different spatial scales to improve object detection performance.

The authors propose a method for scaling concatenation-based models that involves increasing the number of channels in the network while maintaining a constant input resolution. This is intended to increase the expressive power of the network, allowing it to learn more complex and nuanced features that are better suited for object detection.

They also suggest increasing the depth of the network and adjusting the number of feature fusion layers to optimize the trade-off between model complexity and performance.

Overall, the idea of model scaling for concatenation-based models is to fine-tune the architecture of the network to achieve the best possible performance for the given task, taking into account factors such as computational efficiency and memory constraints.

### Re-Parametrization Planning

Re-Parametrization Planning is used, which consists in a technique for optimizing neural network architectures to improve efficiency and reduce redundancy, while maintaining or improving performance on the target task.

In traditional convolutional layers, each filter has a separate set of weights and biases for each input channel. This can lead to a large number of redundant parameters, particularly in deeper networks with many channels.

The authors propose a new type of convolutional layer that uses shared weights and biases across multiple channels. Specifically, they use a factorized weight matrix, where the filter weights are decomposed into two smaller matrices, one for the input channels and one for the output channels. This reduces the number of parameters needed to represent the convolutional filters, while still allowing the network to learn rich and diverse features.

The authors also introduce a method for dynamically adjusting the number of channels in the network based on the size of the input image. This allows the network to be more efficient when processing images of different sizes, without sacrificing accuracy.

![YOLO v7 Architecture: Re-Parametrization Planning](./pics/yolo_v7_architecture_4.jpg)


### Auxiliary Head

YOLO v7 has a multi-headed architecture; in multi-head networks there is a separate head for each object class. Each head consists of a set of convolutional layers that process the features extracted by the backbone of the network, and then output the predicted bounding boxes and class probabilities for objects of that class.

In YOLOv7, the authors use a variant of this approach called "context-based multi-head," where the heads are organized according to the context of the object classes. This means that the heads for related object classes are grouped together, allowing them to share features and improve the accuracy of the predictions.

Additionally, **auxiliary heads** are used during **training**: smaller and shallower sub-network that are trained to predict intermediate features of the input image. The idea behind the auxiliary head is to provide additional supervision to the network during training, which can help to prevent overfitting and improve generalization performance.

During training, the loss function of the network is computed using both the predictions from the main head and the auxiliary head. The loss from the auxiliary head is weighted less than the loss from the main head, since the primary focus is still on predicting the bounding boxes and class probabilities.

At inference time, the auxiliary head is discarded, and only the main head is used to predict the final object detections.

![YOLO v7 Architecture: Auxiliary Head](./pics/yolo_v7_architecture_5.jpg)


## 3. Custom Dataset

### Capturing a Custom Dataset with OpenCV

```python

```

### Finding a Public Dataset


### Labeling on Roboflow

### Augmentation

### Exporting the Dataset

## 4. Training

## 5. Deployment

