import numpy as np
import logging
import xml.etree.ElementTree as ET
import cv2
import os

from torch.utils.data import Dataset


class VOCDetection(Dataset):
	bbox_format = 'pascal_voc'

	def __init__(self, root, year=None, image_set='train', label_file=None,
	             transform=None, keep_difficult=False):
		"""Dataset for VOC data.
		Args:
			root: the root of the VOC-style dataset, the directory contains the
		following sub-directories: Annotations, ImageSets, JPEGImages,
		SegmentationClass, SegmentationObject.
		"""
		super(VOCDetection, self).__init__()

		self.transform = transform

		if year is not None:
			infix = "VOC%s" % year
		else:
			infix = ""

		self.root = os.path.join(root, infix)

		image_sets_file = os.path.join(self.root, "ImageSets", "Main",
		                               "%s.txt" % image_set)

		self.ids = self._read_image_ids(image_sets_file)
		self.keep_difficult = keep_difficult

		# if the labels file exists, read in the class names
		if label_file is None:
			label_file = os.path.join(self.root, "labels.txt")

		if os.path.isfile(label_file):
			class_string = ""
			with open(label_file, 'r') as infile:
				for line in infile:
					class_string += line.rstrip()

			# classes should be a comma separated list

			classes = class_string.split(',')
			# prepend BACKGROUND as first class
			classes.insert(0, 'BACKGROUND')
			classes  = [ elem.replace(" ", "") for elem in classes]
			self.class_names = tuple(classes)
			logging.info("VOC Labels read from file: " + str(self.class_names))

		else:
			logging.info("No labels file, using default VOC classes.")
			self.class_names = ('BACKGROUND',
			'aeroplane', 'bicycle', 'bird', 'boat',
			'bottle', 'bus', 'car', 'cat', 'chair',
			'cow', 'diningtable', 'dog', 'horse',
			'motorbike', 'person', 'pottedplant',
			'sheep', 'sofa', 'train', 'tvmonitor')

		self.class_dict = {
			class_name: i for i, class_name in enumerate(self.class_names) }

	def __getitem__(self, index):
		image_id = self.ids[index]
		boxes, labels, is_difficult = self._get_annotation(image_id)

		if not self.keep_difficult:
			boxes = [b for i, b in enumerate(boxes) if not is_difficult[i]]
			labels = [l for i, l in enumerate(labels) if not is_difficult[i]]

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

	@staticmethod
	def _read_image_ids(image_sets_file):
		ids = []
		with open(image_sets_file) as f:
			for line in f:
				ids.append(line.rstrip())
		return ids

	def _get_annotation(self, image_id):
		annotation_file = os.path.join(self.root, "Annotations",
		                               f"{image_id}.xml")
		objects = ET.parse(annotation_file).findall("object")

		boxes = []
		labels = []
		is_difficult = []

		for object in objects:
			class_name = object.find('name').text.lower().strip()
			# we're only concerned with classes in our list
			if class_name in self.class_dict:
				bbox = object.find('bndbox')

				# VOC dataset format follows Matlab,
				# in which indexes start from 1
				x1 = float(bbox.find('xmin').text) - 1
				y1 = float(bbox.find('ymin').text) - 1
				x2 = float(bbox.find('xmax').text) - 1
				y2 = float(bbox.find('ymax').text) - 1
				boxes.append([x1, y1, x2, y2])

				labels.append(self.class_dict[class_name])
				is_difficult_str = object.find('difficult').text
				is_difficult.append(int(is_difficult_str)
				                    if is_difficult_str else 0)

		return boxes, labels, is_difficult

	def _read_image(self, image_id):
		image_file = os.path.join(self.root, "JPEGImages", f"{image_id}.jpg")
		image = cv2.imread(str(image_file))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image
