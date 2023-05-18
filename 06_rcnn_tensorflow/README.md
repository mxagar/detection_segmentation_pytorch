# PyImageSearch: R-CNN with Tensorflow

This mini-project is a series of 4 tutorials in which a Region Proposal CNN is built from scratch. Both theory and practical aspects are visited.

These are the related blog posts:

1. [Turning Any Deep Learning Image Classifier into an Object Detector](https://pyimagesearch.mykajabi.com/products/pyimagesearch-university-full-access-plan/categories/4727361/posts/2147742271)
2. [OpenCV Selective Search for Object Detection](https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/?_ga=2.24076154.1576899293.1684317349-844635163.1684131075)
3. C
4. D

The blog posts show intuitively how we can go from classifiers to object detectors, ending up in a commonly used Region Proposal CNN, or R-CNN.

## 1. Turning any CNN image classifier into an object detector with Keras, TensorFlow, and OpenCV

Material:

- [Blog post](https://www.pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/?_ga=2.36459488.1576899293.1684317349-844635163.1684131075)
- [Google Colab](https://colab.research.google.com/drive/1MrgG83e-XGSEKlVLrHB7gT1tJ5PHeu-m?usp=sharing)
- [Code](https://pyimagesearch-code-downloads.s3-us-west-2.amazonaws.com/classifier-to-detector/classifier-to-detector.zip)

In this section/mini-project a pre-trained CNN (or any classifier) is converted into an object detector using **image pyramids** and **sliding windows**. The first allow for a multi-scale representation on which ROIs of different sizes are slided; for each capture image patch, a classifier is run (a CNN, linear SVM, etc.).

As the window gets closer to a learned object, the probabilities increase; up from a threshold, we take the proposal as valid. This approach is similar to what is done by **HOG, Histogram Oriented Gradients** or **Haar Cascades**.

When we use sliding windows, end up with many overlapping regions. We can collapse all these ROIs with **non-maximal supression** applied class-wise (see explanation in main `README.md`).

All this approach is a first approximation with which we can convert any classification network to be an object detection network; however, the resulting model has important limitations:

- It is very slow: we need to generate many image patches and run the classifierin all of them.
- It is constrained to the chosen window size/shape; that could be extended, but we'd require much more time.
- Bouding box locations are not accurate.
- The network is not end-to-end trainable: we train the classifier, but the bounding box detection algorithm is not trained!

![Image Pyramids, from PyImageSearch](./pics/image_pyramids.jpg)

![Sliding Windows, from PyImageSearch](./pics/sliding_window_example.gif)

![Sliding Windows, from PyImageSearch](./pics/sliding_window_approach.jpg)

We can run the notebook [`classifier_to_detector.ipynb`](./classifier-to-detector/classifier_to_detector.ipynb) or the provided python script [`detect_with_classifier.py`](./classifier-to-detector/detect_with_classifier.py):

```bash
conda activate ds
cd classifier-to-detector

# Script
python detect_with_classifier.py --image images/hummingbird.jpg --size "(250, 250)"

# Notebook
jupyter lab classifier_to_detector.ipynb
```

