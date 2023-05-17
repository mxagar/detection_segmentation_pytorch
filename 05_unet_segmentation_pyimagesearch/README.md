# PyImageSearch: U-Net Segmentation

This folder contains examples/code from the following PyImageSearch tutorial:

[U-Net: Training Image Segmentation Models in PyTorch](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/?_ga=2.73178069.791523268.1684131076-844635163.1684131075)

Everything is implemented in the notebook [`unet_pytorch.ipynb`](./unet_pytorch.ipynb); the code from the notebook is transformed into scripts in the folder [`unet-tgs-salt-pytorch`](./unet-tgs-salt-pytorch).

The project uses the Kaggle dataset [tgs-salt-identification-challenge](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/overview). There, below-earth salt deposits are detected from seismic images; detecting these salt sediments is fundamental for safe oil extraction, among others. The dataset consits of image-mask pairs; the goal is to perform a pixel-wise binary classification, which determines whether a pixel is (i) salt or (ii) non-salt / sediment.

Summary of the contents:

- A custom dataset loader is defined, which optimally loads images & masks when `__getitem__()` is invoked.
- A UNet architeture is defined with all its layers; very interesting implementation.
- A custom data pipeline is defined: dat aloader, tranformers, train/test split, etc.
- Training is performed with validation in each epoch.
- Predictions are done with test images and the results plotted.

Limitations:

- We have only one class / label.
- The images are quite simple.

Material links:

- [Google Colab](https://colab.research.google.com/drive/1qRfXv17pfdvKjZM21b0k8YUp5k-E1M6g?usp=sharing)
- [Code](https://pyimagesearch-code-downloads.s3-us-west-2.amazonaws.com/unet-tgs-salt-pytorch/unet-tgs-salt-pytorch.zip)
- [Blog post](https://www.pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/?_ga=2.95606874.791523268.1684131076-844635163.1684131075)

Dependecies:

```bash
# Create an environment
conda env create -f conda_ds.yaml
conda activate ds

# Otherwise, if we are using any other standard environment
# we can simply install these
pip install torch torchvision
pip install opencv-contrib-python
```

Project structure:

```
.
|-- unet-tgs-salt-pytorch
|   |-- dataset
|   |   `-- train
|   |       |-- images
|   |       |   |-- 000e218f21.png
|   |       |   |-- ...
|   |       |   `-- fff987cdb3.png
|   |       `-- masks
|   |           |-- 000e218f21.png
|   |           |-- ...
|   |           `-- fff987cdb3.png
|   |-- output
|   |   |-- plot.png
|   |   |-- test_paths.txt
|   |   `-- unet_tgs_salt.pth
|   |-- predict.py
|   |-- pyimagesearch
|   |   |-- config.py
|   |   |-- dataset.py
|   |   `-- model.py
|   `-- train.py
`-- unet_pytorch.ipynb
```

To run the project, we need to download the code and the dataset. **BUT, IMPORTANT: First we need to agree to the terms in the Rules section of the competition**: [tgs-salt-identification-challenge/rules](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/rules). Also, more information on the Kaggle API: [Kaggle/kaggle-api](https://github.com/Kaggle/kaggle-api).

```bash
# Get package/code
conda activate ds
wget https://pyimagesearch-code-downloads.s3-us-west-2.amazonaws.com/unet-tgs-salt-pytorch/unet-tgs-salt-pytorch.zip
unzip -qq unet-tgs-salt-pytorch.zip
cd unet-tgs-salt-pytorch

# Get the Kaggel API keys and place them in home 
pip install kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
# Set path
kaggle config set -n path -v ~/.kaggle/
# Check that the path was correctly set
kaggle config view

# Download the dataset
# IMPORTANT: accept the rules/terms on the Kaggle web interface
# https://www.kaggle.com/competitions/tgs-salt-identification-challenge/rules
kaggle competitions download -c tgs-salt-identification-challenge -f train.zip
mkdir -p "dataset/train"
cp ~/.kaggle/competitions/tgs-salt-identification-challenge/train.zip dataset/train
cd dataset/train
unzip -qq train.zip
rm train.zip
cd ../..
pwd # unet-tgs-salt-pytorch

# Option 1: Open and run the jupyter notebook
# Same code as in the scripts, but organized in modules + MY NOTES
jupyter notebook unet_pytorch.ipynb

# Option 2: Run ready scripts
python train.py
python predict.py
```

