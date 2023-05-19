# PyImageSearch: R-CNN with Tensorflow

This mini-project is a series of 4 tutorials in which a Region Proposal CNN is built from scratch. Both theory and practical aspects are visited. We see intuitively how we can go from classifiers to object detectors, passing thorough different stages.

These are the related blog posts:

1. [Turning Any Deep Learning Image Classifier into an Object Detector](https://pyimagesearch.mykajabi.com/products/pyimagesearch-university-full-access-plan/categories/4727361/posts/2147742271)
2. [OpenCV Selective Search for Object Detection](https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/?_ga=2.24076154.1576899293.1684317349-844635163.1684131075)
3. C
4. D

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

We can run the notebook [`classifier_to_detector.ipynb`](./classifier_to_detector.ipynb) or the provided python script [`detect_with_classifier.py`](./classifier-to-detector/detect_with_classifier.py):

```bash
conda activate ds
cd classifier-to-detector

# Script
python detect_with_classifier.py --image images/hummingbird.jpg --size "(250, 250)"

# Notebook
cd ..
jupyter lab classifier_to_detector.ipynb
```

## 2. OpenCV Selective Search for Object Detection

Material:

- [Blog post](https://www.pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/?_ga=2.258874826.1576899293.1684317349-844635163.1684131075)
- [Google Colab](https://colab.research.google.com/drive/1irBEGter1EPoPTIIMrNkJKMWLeZZhOFP?usp=sharing)
- [Code](https://pyimagesearch-code-downloads.s3-us-west-2.amazonaws.com/opencv-selective-search/opencv-selective-search.zip)

This mini-project starts adressing the limitations of the previous approach: (1) slow computation due to a search in a large space and (2) fixed window shape. To that end, **selective search** is introduced: basically, we use classical algorithms on the image to detect probably interesting ROIs which are passed to the CNN, instead of sliding windows of different sizes, which is a brute force apprach.

The [Selective Search](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf) algorithm was proposed in 2012 and it can replace any pryramid+sliding algorithm, being much faster. It oversegments an image and takes those small segments are ROIs.

Oversegmentation is like a tesselation of an image based on

- Color similarity.
- Texture similarity.
- Size similarity. 
- Shape similarity
- A final meta-similarity, which is a linear combination of the above similarity measures

The idea is that we compute [super-pixels](https://pyimagesearch.com/tag/superpixel/) on an image (similar cells) and then build a hirarchy upwards depending on cell similarities.

- Color similarity: channel histograms are compared.
- Texture similarity: derivatives in different orientations are compared.
- Size similarity: the merging of smaller regions is prioritized.
- Shape similarity: gap filling.
- A final meta-similarity, which is a linear combination of the above similarity measures

As we build the hierarchy, objects start to appear in the image; effectively, we pass intermediate clusters to the network. Those clusters are contained by proposed ROIs, which are used to crop image patches passed to the CNN.

![Selective Serach, from PyImageSearch](./pics/selective_search.jpg)

Thus, selective search is similar to [saliency detection](https://pyimagesearch.com/2018/07/16/opencv-saliency-detection/): regions are identified which could contain objects.

The contributions package of OpenCV has a built-in module which performs automatically selective search; that's what is shown in the code.

We can run the notebook [`opencv_selective_search.ipynb`](./opencv_selective_search.ipynb) or the provided python script [`selective_search.py`](./opencv-selective-search/selective_search.py):

```bash
conda activate ds
cd classifier-to-detector

# Script
python selective_search.py --image dog.jpg 

# Notebook
cd ..
jupyter lab opencv_selective_search.ipynb
```

Interesting links:

- [Selective Search poster](https://www.koen.me/research/pub/vandesande-iccv2011-poster.pdf)
- [Saliency detection](https://pyimagesearch.com/2018/07/16/opencv-saliency-detection/)
