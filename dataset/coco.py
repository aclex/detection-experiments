import numpy as np
import logging
import cv2
import os

from torch.utils.data import Dataset


class CocoDetection(Dataset):
	bbox_format = 'coco'

	def __init__(self, root, ann_file, transform=None):
		"""Dataset for COCO data.
		Args:
			root: the root of the COCO-style dataset where images are stored
			ann_file: JSON file with COCO-style dataset annotations
		"""
		super(CocoDetection, self).__init__()

		self.root = root
		self.transform = transform

		from pycocotools.coco import COCO
		self.coco = COCO(self._get_abs_path(ann_file))
		self.ids = list(sorted(self.coco.imgs.keys()))

		categories = self.coco.cats

		self.class_names = ['BACKGROUND']
		self.class_ids = [-1]

		for cat_id, cat_name in categories.items():
			self.class_names.append(cat_name["name"])
			self.class_ids.append(cat_id)

	def __getitem__(self, index):
		image_id = self.ids[index]
		boxes, labels = self._get_annotation(image_id)

		image = self._read_image(image_id)

		result = {
			"image": image,
			"bboxes": boxes,
			"category_id": labels
		}

		if self.transform:
			result = self.transform(**result)

		return result

	def get_image(self, index):
		image_id = self.ids[index]
		image = self._read_image(image_id)

		if self.transform:
			image, _ = self.transform(image=image)

		return image

	def get_annotation(self, index):
		image_id = self.ids[index]
		return image_id, self._get_annotation(image_id)

	def __len__(self):
		return len(self.ids)

	def _get_annotation(self, image_id):
		ann_ids = self.coco.getAnnIds(imgIds=image_id)
		objects = self.coco.loadAnns(ann_ids)

		boxes = []
		labels = []

		for object in objects:
			boxes.append(object['bbox'])
			labels.append(self.class_ids.index(object['category_id']))

		return boxes, labels

	def _get_abs_path(self, path):
		if os.path.isabs(path):
			return path
		else:
			return os.path.join(self.root, path)

	def _read_image(self, image_id):
		image_file = self._get_abs_path(
			self.coco.loadImgs(image_id)[0]['file_name'])

		image = cv2.imread(str(image_file))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		return image
