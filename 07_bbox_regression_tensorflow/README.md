# PyImageSearch: Bounding Box Regression with Tensorflow

This mini-project deals with bounding box regression for 1 or multiple classes; in both cases, though, only one object is detected in the image. These are the associated blog posts:

1. [Object detection: Bounding box regression with Keras, TensorFlow, and Deep Learning](https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/?_ga=2.164853631.1777097551.1684564539-1295398344.1684564539)
2. [Multi-class object detection and bounding box regression with Keras, TensorFlow, and Deep Learning](https://pyimagesearch.com/2020/10/12/multi-class-object-detection-and-bounding-box-regression-with-keras-tensorflow-and-deep-learning/?_ga=2.67263313.1777097551.1684564539-1295398344.1684564539)

## 1. Object detection: Bounding box regression with Keras, TensorFlow, and Deep Learning

Material links:

- [Blog post](https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/?_ga=2.164853631.1777097551.1684564539-1295398344.1684564539)
- [Code](https://pyimagesearch-code-downloads.s3-us-west-2.amazonaws.com/bounding-box-regression/bounding-box-regression.zip)
- [Google Colab](https://colab.research.google.com/drive/1vQ587yUumW0xEEPPj-6aYUPi1eYjshs2?usp=sharing)

This is a very simple blog post in which a 4D regression is added to a pre-trained network (VGG) to predict the bounding box of planes on images. The regressor has 3.2M parameters.

The dataset is the [CALTECH-101](https://data.caltech.edu/records/mzrjq-6wc02), consisting of 800 airplane images with their bounding boxes (in a CSV).

The project is very limited:

- Only one object class.
- Only one bounding box instance in the image.

We can run the notebook [`bounding_box_regression.ipynb`](./bounding_box_regression.ipynb) or the provided python scripts in [`bounding-box-regression`](./bounding-box-regression).

## 2. Multi-class object detection and bounding box regression with Keras, TensorFlow, and Deep Learning

Material links:

- [Blog post](https://www.pyimagesearch.com/2020/10/12/multi-class-object-detection-and-bounding-box-regression-with-keras-tensorflow-and-deep-learning/?_ga=2.72423251.1777097551.1684564539-1295398344.1684564539)
- [Code](https://pyimagesearch-code-downloads.s3-us-west-2.amazonaws.com/multi-class-bbox-regression/multi-class-bbox-regression.zip)
- [Google Colab](https://colab.research.google.com/drive/1U8N3pJPIHzRuhhZ8K2Rxh0wmCgftVPEb?usp=sharing)

This tutorial is also very simple, but educationally interesting; it is the equivalent of [`04_basic_object_detection_pyimagesearch/02_trained/object_detector_in_pytorch.ipynb`](../04_basic_object_detection_pyimagesearch/02_trained/object_detector_in_pytorch.ipynb), but implemented in Tensoflow/Keras.

Architeture description:

- The pre-trained VGG16 network is used as backbone, with weights frozen.
- To the backbone, we attach two networks (*heads*) in parallel: one for regressing the bounding boxes, one for the classification labels.
- We assume we have only one object in an image; that's an important limitation, but the motivation of the network is educational; the model is defined and trained from-scratch using pre-trained weights of the backbone.
- Overall I thinks it's a nice example to see how to implement things end-to-end, but we need to be aware of the **limitations**:
  - We have 3 custom objects, but:
    - The annotations are provided already.
    - The objects are quite similar to some COCO/ImageNet classes, nothing really weird: airplane, face, motorcycle.
  - The data is passed as a huge array; that could be improved by openening images only when needed.
  - Only one object is detected in an image.

We can run the notebook [`multi_class_bbox_regression.ipynb`](./multi_class_bbox_regression.ipynb) or the provided python scripts in [`multi-class-bbox-regression`](./multi-class-bbox-regression/).
