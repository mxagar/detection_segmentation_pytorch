# YOLO v3 with Darknet

YOLO is a state-of-the-art, real-time object detection algorithm. In this notebook, the YOLO model is applied to detect objects in images.

The code from this notebook has its origin in the Udacity repository [deep-learning-v2-pytorch](https://github.com/mxagar/deep-learning-v2-pytorch).

Note that YOLO uses originally [Darknet](https://pjreddie.com/darknet/). As the notebook says:

> "The version of Darknet used in this notebook has been modified to work in PyTorch 0.4 and has been simplified because we won't be doing any training. Instead, we will be using a set of pre-trained weights that were trained on the Common Objects in Context (COCO) database."

Therefore, in this example we can simply run YOLOv3 to predict the COCO object classes -- no new classes of objects can be used straightforwardly.
