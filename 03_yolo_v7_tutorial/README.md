# Custom Object Detection with YOLO v7

I made these notes following these courses/tutorials:

- [Nicolai Nielsen: YOLOv7 Custom Object Detection](https://nicolai-nielsen-s-school.teachable.com/courses)
- [Youtube: YOLO Object Detection Models, Nicolai Nielsen](https://www.youtube.com/playlist?list=PLkmvobsnE0GEfcliu9SXhtAQyyIiw9Kl0)
- [Original yolov7 implementation repository](https://github.com/WongKinYiu/yolov7)
- [Fine Tuning YOLOv7 on Custom Dataset](https://learnopencv.com/fine-tuning-yolov7-on-custom-dataset/)
- [YOLOv7 Object Detection Paper Explanation & Inference](https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/)

In addition to the content in those tutorials, I compiled information from other sources, as the YOLO v7 paper and repository.

Table of contents:

- [Custom Object Detection with YOLO v7](#custom-object-detection-with-yolo-v7)
  - [1. Introduction](#1-introduction)
  - [2. YOLOv7 Architecture](#2-yolov7-architecture)
    - [Efficient Layer Aggregation](#efficient-layer-aggregation)
    - [Model Scaling Techniques](#model-scaling-techniques)
    - [Re-Parametrization Planning](#re-parametrization-planning)
    - [Auxiliary Head](#auxiliary-head)
  - [3. Getting a Custom Dataset](#3-getting-a-custom-dataset)
    - [Capturing a Custom Dataset with OpenCV](#capturing-a-custom-dataset-with-opencv)
    - [Finding a Public Dataset](#finding-a-public-dataset)
    - [Labelling on Roboflow](#labelling-on-roboflow)
    - [Generate the Dataset: Preprocessing and Augmentation](#generate-the-dataset-preprocessing-and-augmentation)
    - [Exporting the Dataset](#exporting-the-dataset)
    - [Roboflow Model Training](#roboflow-model-training)
  - [4. YOLO v7 Repository: Notes](#4-yolo-v7-repository-notes)
  - [5. Application Notebook](#5-application-notebook)
    - [1. Install Dependencies and Set Up Environment](#1-install-dependencies-and-set-up-environment)
      - [GPU Testing](#gpu-testing)
      - [YOLOv7 Repository + Roboflow](#yolov7-repository--roboflow)
    - [2. Download Custom Dataset from Roboflow](#2-download-custom-dataset-from-roboflow)
      - [Get the Dataset](#get-the-dataset)
      - [Check the Dataset Structure](#check-the-dataset-structure)
    - [3. Training](#3-training)
      - [Get Pre-Trained Weights](#get-pre-trained-weights)
      - [Train](#train)
    - [4. Test Results](#4-test-results)
    - [5. Export the Model](#5-export-the-model)
    - [6. Deployment](#6-deployment)
      - [Roboflow Static Deployment: API](#roboflow-static-deployment-api)
      - [Pytorch Online Webcam Deployment](#pytorch-online-webcam-deployment)
      - [ONNX Deployment](#onnx-deployment)

## 1. Introduction

The original YOLO v7 paper is implemented in the following repository:

[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

The code in that repository is used, after cloning it to [`lab/`](./lab/). Also, the pre-trained weights from that repo are used.

General requirements:

- [Roboflow](https://roboflow.com) account
- Anaconda
- Google Colab / Jupyter
- Python 3.9
- PyTorch (the newest version on Google Colab)
- OpenCV (all versions can be used, e.g., 4.5.2)
- Onnx runtime

Basic setup:

```bash
# Crate env: requirements in conda.yaml
# This packages are the basic for Pytorch-CUDA usage
# HOWEVER, we need to install more later on
# when we clone and use the  original YOLO v7 repo
conda env create -f conda.yaml
conda activate yolov7

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

# Pytorch: Mac / Windows CPU (not tested)
python -m pip install torch torchvision torchaudio

# Dump installed libraries in pip format
python -m pip list --format=freeze > requirements.txt
```

In case of any runtime errors, check the torch-torchvision compatibility table: [Installation](https://github.com/pytorch/vision#installation).

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

In the following, the most important architectural properties/elements are discussed.

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

## 3. Getting a Custom Dataset

This tutorial uses [Roboflow](https://roboflow.com) to generate a custom dataset; however, any labeler could be used if the required dataset format is followed; more on the dataset format in the application notebook (see section [5. Application Notebook](#5-application-notebook)).

### Capturing a Custom Dataset with OpenCV

The file [`record_images.py`](./lab/record_images.py) captures with the system camera images of a scene while we press `S`. We should place 4-5 objects on our desk and capure around 100 images (saved in [`./lab/data/captured/`](./lab/data/captured/)) while moving the camera/objects.

```python
import cv2

# Set camera: 0 default/laptop, 1 webcam, etc.
cap = cv2.VideoCapture(1)
# Initialize
num = 0
path = "data/captured"

while cap.isOpened():
    succes1, img = cap.read()
    k = cv2.waitKey(1)

    if k == 27: # ESC
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(path + '/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv2.imshow('Img '+str(num), img)

cap.release()
cv2.destroyAllWindows()
```

I captured 35 images of three objects: a cup, a ball and a baby lotion toy. The images have a size of `(480, 640, 3)`.

![Scenario with 3 objects: cup, ball, lotion](./assets/scenario.png)

### Finding a Public Dataset

We label the images using [Roboflow](https://roboflow.com). In that platform, it's possible to find public datasets once we log in:

    Universe > Object Detection: Select one dataset

For instance: [sakis2](https://universe.roboflow.com/project-coewi/sakis2/dataset/2).

We can:

- Browse the images and labels
- Use the loaded models, if any
- Download the dataset: as a ZIP or with an URL

When we select download, we need to select the label format; for the tutorial we use `TXT YOLOv7 Pytorch`.

If we choose the URL, we get a snippet like this, which allows for downloading the dataset to a local folder.

```python
!pip install roboflow

# Get API key:
# Projects > Settings > Roboflow API: Private API Key, Show
# Do not publish this key
# Alternatively, persist in local file, don't commit,
# and load from file
with open('roboflow.key', 'r') as file:
    api_key = file.read().strip()

# Download dataset to local folder
from roboflow import Roboflow
rf = Roboflow(api_key=api_key)
project = rf.workspace("project-coewi").project("sakis2")
# This will download to the local folder the dataset directory
# with the specified format
dataset = project.version(2).download("yolov7")
```

We can download the ZIP or via code, and then create a new project of our own and upload a dataset!

    Roboflow > Projects > Create new > Upload folder
      Car Detection

### Labelling on Roboflow

We can create a project and upload our images:

    Roboflow > Projects > Create new > Upload folder

After uploading the images, we need to annotate them:

- We assign a person (me)
- We start annotating with the interface

![Roboflow Annotation Interface](./pics/roboflow_interface.jpg)

The Roboflow interface has these menus/pages:

    Universe Page / Upload / Assign / Annotate / Dataset / Generate / Versions / Deploy / Health Check

During annotation, we have the following tools:

- Hand, pick
- Box: we add a class name if new, select if already defined
- Polygon
- Smart polygon: automatic polygon segmentation
- Auto-labelling with existing pre-trained models, if the object classes are defined in, e.g., COCO
- Repeat previous: for video frames
- etc.

Then, we go back and `Add Images to Dataset`. The split is created.

We can check the `Health Check`, which gives relevant information on the dataset:

- Histograms of class instances
- Heatmaps
- Size information
- Several statistics
- etc.

### Generate the Dataset: Preprocessing and Augmentation

After we annotate the images, we need to explicitly **generate** the dataset. For that, we can add two steps:

- Preprocessing: Auto-Orient, Resize, etc.
- Augmentation: we have many image/bbox-level effects:
  - Flip
  - Rotate
  - Crop

If data augmentation and preprocessing are chosen, new images are generated, which can be downloaded later on. Thus, we don't need to create data loaders with transformers in the code.

### Exporting the Dataset

Once the dataset has been generated, we can train a model on Roboflow! However, we instead download the dataset and train our YOLOv7 model.

Under `Versions`, we select the version and `Export`.

- Select format: TXT YOLO v7
- Select python snippet

```python
rf = Roboflow(api_key=api_key)
project = rf.workspace("mikel-sagardia-tknfd").project("basic-object-detection-qkmda")
dataset = project.version(1).download("yolov7")
```

### Roboflow Model Training

Roboflow trains automatically a model that best fits to the dataset we have uploaded and labelled. Later on, we can access it via the Roboflow API as follows:

```python
model = project.version(1).model
```

The notebook [`YOLOv7Application.ipynb`](./lab/YOLOv7Application.ipynb) shows how to use it.

## 4. YOLO v7 Repository: Notes

The original YOLO v7 paper is implemented in the following repository:

[WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

The code in that repository is used, after cloning it to [`lab/`](./lab/). Also, the pre-trained weights from that repo are used.

I should probably spend some time having a look at the code, since it seems to be nicely done.

The repository has custom scripts that do all the job:

- [`train.py`](./lab/yolov7/train.py)
- [`test.py`](./lab/yolov7/test.py)
- [`detect.py`](./lab/yolov7/detect.py)

There are also many auxiliary scripts and tools for anything:

- [`models/yolo.py`](./lab/yolov7/models/yolo.py)
- `cfg`
- `tools`
- `utils`
- etc.

It seems that they use the Danknet configuration files, which are parsed and used to build the YOLO model with PyTorch.

## 5. Application Notebook

In the original repository, the authors explain how to use their scripts for training and inference. The application notebooks in this tutorial build around the guidelines provided in that original repository.

The notebook [`Setup.ipynb`](./lab/Setup.ipynb) shows how to inspect our GPUs.

The notebook [`YOLOv7Application.ipynb`](./lab/YOLOv7Application.ipynb) contains the entire YOLO training and testing application; additionally, for deployment, these files can be used:

- [`detect_roboflow.py`](./lab/detect_roboflow.py)
- [`detect_pytorch_webcam_opencv.py`](./lab/yolov7/detect_pytorch_webcam_opencv.py)
- [`detect_onnx_webcam_opencv.py`](./lab/detect_onnx_webcam_opencv.py)

In the following, the contents from [`YOLOv7Application.ipynb`](./lab/YOLOv7Application.ipynb) are pasted; it is assumed, we are running on a Windows system with a custom conda environment installed as explained in the [Introduction](#1-introduction).

### 1. Install Dependencies and Set Up Environment

See the [Introduction](#1-introduction).

#### GPU Testing

I run the notebook on a Windows with an exteral GPU: NVIDIA RTX 3036.

```python
# Check GPUs
!nvidia-smi

# Check that the GPU can be connected/accessed correctly
import os
import torch
import torchvision

# If you are running this notebook locally
# set environment variable with possible device ids
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(os.environ["CUDA_VISIBLE_DEVICES"])
# Set device: 0 or 1
# NOTE: indices are not necessarily the ones shown by nvidia-smi
# We need to try them with the cell below
torch.cuda.set_device("cuda:0")

# Check that the selected device is the desired one
print("Torch version?", torch.__version__)
print("Torchvision version?", torchvision.__version__)
print("Is cuda available?", torch.cuda.is_available())
print("Is cuDNN version:", torch.backends.cudnn.version())
print("cuDNN enabled? ", torch.backends.cudnn.enabled)
print("Device count?", torch.cuda.device_count())
print("Current device?", torch.cuda.current_device())
print("Device name? ", torch.cuda.get_device_name(torch.cuda.current_device()))

x = torch.rand(5, 3)
print(x)
# Torch version? 1.13.0+cu117
# Torchvision version? 0.14.0+cu117
# Is cuda available? True
# Is cuDNN version: 8500
# cuDNN enabled?  True
# Device count? 2
# Current device? 0
# Device name?  NVIDIA GeForce RTX 3060
# tensor([[0.6014, 0.5259, 0.4306],
#         [0.8409, 0.8252, 0.6665],
#         [0.6248, 0.7593, 0.3523],
#         [0.7337, 0.7629, 0.9790],
#         [0.7688, 0.3467, 0.4424]])
```

#### YOLOv7 Repository + Roboflow

```python
# Download/clone YOLOv7 repository and install its (additional) requirements
!git clone https://github.com/WongKinYiu/yolov7
%cd yolov7
!pip install -r requirements.txt

# Additional requirements
!pip install roboflow
```

### 2. Download Custom Dataset from Roboflow

```python
# Note we are inside the YOLO repo folder
# We dowload our ROboflow dataset there
%pwd
# 'C:\\Users\\Mikel\\git_repositories\\detection_segmentation_pytorch\\03_yolo_v7_tutorial\\lab\\yolov7'
```

#### Get the Dataset

```python
# Get API key:
# Projects > Settings > Roboflow API: Private API Key, Show
# Do not publish this key
# Alternatively, persist in local file, don't commit,
# and load from file
with open('../roboflow.key', 'r') as file:
    api_key = file.read().strip()

from roboflow import Roboflow
rf = Roboflow(api_key=api_key)
project = rf.workspace("mikel-sagardia-tknfd").project("basic-object-detection-qkmda")
dataset = project.version(1).download("yolov7")
```

#### Check the Dataset Structure

The Roboflow `dataset` object contains several information on the dataset. We will pass the `dataset.location/data.yaml` file to the `train.py` scrip, which contains the following:

```yaml
names:
- ball
- cup
- lotion
nc: 3
roboflow:
  license: CC BY 4.0
  project: basic-object-detection-qkmda
  url: https://universe.roboflow.com/mikel-sagardia-tknfd/basic-object-detection-qkmda/dataset/1
  version: 1
  workspace: mikel-sagardia-tknfd
test: ../test/images
train: Basic-Object-Detection-1/train/images
val: Basic-Object-Detection-1/valid/images
```

Apparently, the `roboflow` key is not that important, but the rest of the keys are: `names`, `nc` (number of classes), `test`, `train`, `val`; thus, we can create a similar dataset strucure with another tool than Roboflow. In particular, the image split folder contain:

- unique image names for image files, e.g.,: `img_xxx.jpg`
- a folder `labels` with a TXT for each unique image: `img_xxx.txt`

The `img_xxx.txt` files contain a list of bounding boxes in the image:

```
2 0.56015625 0.4421875 0.140625 0.26875
1 0.5234375 0.76875 0.340625 0.4625
0 0.8421875 0.60625 0.2296875 0.28125
```

The format is the following (coordinates scaled to the image size):

```
class_id x_center, y_center, width, height
```

Thus, the dataset structure, is the following:

```
dataset/
├── test/
│   ├── images/
│   │   └── img_XX1.jpg
│   ├── blabels/
│        └── img_XX1.txt
├── train/
└── val/
```

**BUT**, in another example shown in this [blog post by LearnOpenCV](https://learnopencv.com/fine-tuning-yolov7-on-custom-dataset/), the dataset structure seems to be different:

```
dataset/
├── images
│   ├── test/
│   │   └── img_XX1.jpg
│   ├── train/
│   └── valid/
└── labels
    ├── test/
    │   └── img_XX1.txt
    ├── train/
    └── valid/
```

### 3. Training

#### Get Pre-Trained Weights

```python
# Note we are still in the YOLO repo folder
# We downloaded the pre-trained weights there
%pwd
# 'C:\\Users\\Mikel\\git_repositories\\detection_segmentation_pytorch\\03_yolo_v7_tutorial\\lab\\yolov7'

# Download COCO starting checkpoint f the YOLO model to fine tune
# We are going to upload these weights to fine tune with the training script
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
```

#### Train

```python
import os
import sys
# This variable switches off a warning of OpenMP
if sys.platform == "win32":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

# TRAINING SCRIPT
# NOTES:
# - On Windows, unfortunately we don't have the realtime output
#     Possible fix: https://discourse.jupyter.org/t/how-to-show-the-result-of-shell-instantly/8947/10
# - Before running, we need to log in to wandb
#     >> wandb login
# - The fine-tuned model will be saved by default to
#     runs/train/expXX/weights/best.pt
# - The training metrics and parameters/config are in
#     runs/train/expXX
# - Check all the possible arguments of train.py!
# - We get a wandb experiment tracking report link, check it!
!python train.py --batch 16 \
                 --workers 8 \
                 --epochs 500 \
                 --data {dataset.location}/data.yaml \
                 --weights 'yolov7_training.pt' \
                 --img 640 640 \
                 --device 0 \
                 --cfg cfg/training/yolov7.yaml
```

### 4. Test Results

```python
# DETECT SCRIPT
# Notes:
# - We are passing the dataset test folder
# - Use the desired trained model path, e.g.
#     runs/train/expXX/weights/best.pt
# - The detected images & labels will be saved by default to
#     runs/detect/expXX/
# - Check all possible arguments of detect.py!
!python detect.py --weights runs/train/exp14/weights/best.pt \
                  --conf 0.4 \
                  --img-size 640 \
                  --source {dataset.location}/test/images \
                  --view-img \
                  --save-txt \
                  --save-conf

# Display inference on ALL test images
import glob
from IPython.display import Image, display

i = 0
limit = 10 # max images to print
# Note: use the path where the test images were saved!
for imageName in glob.glob('runs/detect/exp10/*.jpg'): # assuming JPG
    if i < limit:
        display(Image(filename=imageName))
        print("\n")
    i = i + 1
```

### 5. Export the Model

If we export the model (currently a Pytorch model) to the standard [ONNX](https://onnx.ai/), we can use it to perform live inference with OpenCV. To that end, we need to install some additional packages.

```python
!pip install --upgrade setuptools pip --user
!pip install protobuf<4.21.3
!pip install onnx>=1.9.0
!pip install onnxruntime
!pip install onnxruntime-gpu
!pip install onnx-simplifier>=0.3.6 --user

# Select the model exp version we'd like to export to ONNX
# The ONNX model is exported to the same folder as the file best.onnx
!python export.py --weights runs/train/exp14/weights/best.pt \
                  --grid \
                  --end2end \
                  --simplify \
                  --topk-all 100 \
                  --iou-thres 0.45 \
                  --conf-thres 0.2 \
                  --img-size 640 640 \
                  --max-wh 640
# For onnxruntime, you need to specify this last value as an integer,
# when it is 0 it means agnostic NMS, otherwise it is non-agnostic NMS
```

### 6. Deployment

Files we need:

- If we want to deploy using the model trained on **Roboflow** (i.e., remotely using the Roboflow API), we just need to access the Roboflow interface, without downloading any model.
- If we want to deploy the **Pytorch** model (locally) we need `.../best.pt`
- If we want to deploy the **ONNX** model (e.g., on locally, OpenCV DNN) we need `.../best.onnx`

#### Roboflow Static Deployment: API

The following code snippet is from [`detect_roboflow.py`](detect_roboflow.py). In it, we basically connect to our Roboflow project model and use the API to perform an inference on an image.

Additionally, although it is not implemented here, we can perform **active learning**:

- We track our inferences: we check the values of the confidences.
- If a predicted confidence is inside a predefined range (e.g., very low, high, etc.), we upload the image to the project dataset: `project.upload()`.
- Then, in the Roboflow web UI, we can label the uploaded images.
- That way, our model becomes better.

More on active learning: [Implementing Active Learning](https://help.roboflow.com/guides/implementing-active-learning).

```python
%cd ..
%pwd
# 'C:\\Users\\Mikel\\git_repositories\\detection_segmentation_pytorch\\03_yolo_v7_tutorial\\lab'

from roboflow import Roboflow

# Get API key:
# Projects > Settings > Roboflow API: Private API Key, Show
# Do not publish this key
# Alternatively, persist in local file, don't commit,
# and load from file
with open('roboflow.key', 'r') as file:
    api_key = file.read().strip()

# Download model
rf = Roboflow(api_key=api_key)
project = rf.workspace("mikel-sagardia-tknfd").project("basic-object-detection-qkmda")
# Check in the Roboflow web UI rge model version we'd like
# This is a Roboflow model object, which in reality points to the Roboflow API
model = project.version(1).model

# Infer on a local image
img_url = "yolov7/Basic-Object-Detection-1/test/images/img9_png.rf.c3bea63eb9645df2c0d196d74b1550d5.jpg"
print(model.predict(img_url, confidence=40, overlap=30).json())
# {'predictions': [{'x': 122.5, 'y': 499.5, 'width': 201.0, 'height': 255.0, 'confidence': 0.902337908744812, 'class': 'ball', 'image_path': 'yolov7/Basic-Object-Detection-1/test/images/img9_png.rf.c3bea63eb9645df2c0d196d74b1550d5.jpg', 'prediction_type': 'ObjectDetectionModel'}, {'x': 379.0, 'y': 361.5, 'width': 140.0, 'height': 203.0, 'confidence': 0.8436849117279053, 'class': 'lotion', 'image_path': 'yolov7/Basic-Object-Detection-1/test/images/img9_png.rf.c3bea63eb9645df2c0d196d74b1550d5.jpg', 'prediction_type': 'ObjectDetectionModel'}, {'x': 119.5, 'y': 256.0, 'width': 179.0, 'height': 282.0, 'confidence': 0.7519652843475342, 'class': 'cup', 'image_path': 'yolov7/Basic-Object-Detection-1/test/images/img9_png.rf.c3bea63eb9645df2c0d196d74b1550d5.jpg', 'prediction_type': 'ObjectDetectionModel'}], 'image': {'width': '640', 'height': '640'}}

# Visualize/save the prediction
model.predict(img_url, confidence=40, overlap=30).save("test_prediction.jpg")

# Infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())

# This is a Roboflow model object, which in reality points to the Roboflow API
print(model)
# {
#   "id": "basic-object-detection-qkmda/1",
#   "name": "Basic Object Detection",
#   "version": "1",
#   "classes": null,
#   "overlap": 30,
#   "confidence": 40,
#   "stroke": 1,
#   "labels": false,
#   "format": "json",
#   "base_url": "https://detect.roboflow.com/"
# }
```

#### Pytorch Online Webcam Deployment

The custom-made file [`yolov7/detect_pytorch_webcam_opencv.py`](yolov7/detect_pytorch_webcam_opencv.py) is a modified version of the original [`detect.py`](yolov7/detect.py) which enables online webcam inference.

The file uses the `best.pt` we want; we access webcam images and show inferences using OpenCV.

I was able to run this script using the NVIDIA eGPU.

**WARNING**: the code needs to cleaning up, refactoring, etc. Notice the different hard-coded parameters.

#### ONNX Deployment

The custom file [`detect_onnx_webcam_opencv.py`](detect_onnx_webcam_opencv.py) is a modified version of the original [`detect.py`](yolov7/detect.py) which enables online webcam inference using the exported ONNX model.

The file uses the `best.onnx` we want; we access webcam images and show inferences using OpenCV, and the inference itself is done with OpenCV, too, leveraging the DNN module.

Unfortunately, I was able to run this script only with the CPU option, which results in a much slower inference (1 FPS). I think that the GPU issue is related to some ONNX dependency/version incompatibility: [CUDAExecutionProvider Not Available](https://github.com/microsoft/onnxruntime/issues/7748).

**WARNING**: the code needs to cleaning up, refactoring, etc. Notice the different hard-coded parameters.