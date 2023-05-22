# PyImageSearch: Bounding Box Regression with Tensorflow

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

