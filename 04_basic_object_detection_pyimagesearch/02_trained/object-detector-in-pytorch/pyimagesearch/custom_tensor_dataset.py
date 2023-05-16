# import the necessary packages
from torch.utils.data import Dataset

class CustomTensorDataset(Dataset):
	# initialize the constructor
	def __init__(self, tensors, transforms=None):
		self.tensors = tensors
		self.transforms = transforms

	def __getitem__(self, index):
		# grab the image, label, and its bounding box coordinates
		image = self.tensors[0][index]
		label = self.tensors[1][index]
		bbox = self.tensors[2][index]

		# transpose the image such that its channel dimension becomes
		# the leading one
		image = image.permute(2, 0, 1)

		# check to see if we have any image transformations to apply
		# and if so, apply them
		if self.transforms:
			image = self.transforms(image)

		# return a tuple of the images, labels, and bounding
		# box coordinates
		return (image, label, bbox)

	def __len__(self):
		# return the size of the dataset
		return self.tensors[0].size(0)