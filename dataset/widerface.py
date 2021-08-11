import cv2
import os

from enum import IntEnum

import torch

from torch.utils.data import Dataset


class WIDERFace(Dataset):
	bbox_format = 'coco'

	def __init__(
			self, root, split='train', transform=None):
		"""Dataset of WIDER Face benchmark.
		Args:
			root: the root of the WIDER-style dataset where images are stored
			split: image set to use (either 'train', 'val' or 'test')
		"""
		super().__init__()

		self.root = root
		self.split = split
		self.transform = transform

		self.class_names = ['BACKGROUND', 'face']
		self.class_ids = [0, 1]

		self._data = []

		self._parse_annotation_file()

	def __getitem__(self, index):
		img_path, (boxes, labels) = self.get_annotation(index)
		image = self._read_image(img_path)

		result = {
			"image": image,
			"bboxes": boxes,
			"category_id": labels
		}

		if self.transform:
			result = self.transform(**result)

		return result

	def get_image(self, index):
		item = self._data[index]
		image = self._read_image(item["img_path"])

		if self.transform:
			image, _ = self.transform(image=image)

		return image

	def get_annotation(self, index):
		item = self._data[index]

		image_id = item["img_path"]
		ann = self._get_annotation(item)

		return image_id, ann

	def __len__(self):
		return len(self._data)

	def _get_annotation(self, item):
		if self.split != "train" and self.split != "val":
			return torch.empty(), torch.empty()

		boxes = item["annotations"]["bbox"].to(dtype=torch.float32)
		boxes = boxes[(boxes[..., 2:] > 0).all(dim=-1)]
		labels = torch.ones((len(boxes)), dtype=torch.int)

		return boxes, labels

	def _get_abs_path(self, path):
		if os.path.isabs(path):
			return str(path)
		else:
			return os.path.join(self.root, path)

	def _filename(self):
		if self.split == "train":
			return "wider_face_train_bbx_gt.txt"
		elif self.split == "val":
			return "wider_face_val_bbx_gt.txt"
		elif self.split == "test":
			return "wider_face_test_filelist.txt"

	def _filepath(self):
		filepath = os.path.join(self.root, "wider_face_split", self._filename())
		filepath = os.path.abspath(os.path.expanduser(filepath))

		return filepath

	def _parse_annotation_file(self):
		if self.split == "train" or self.split == "val":
			self._parse_train_val_annotations_file()
		elif self.split == "test":
			self._parse_test_annotations_file()

	def _parse_train_val_annotations_file(self):
		class Section(IntEnum):
			IMAGE_PATH = 0,
			OBJECT_COUNT = 1,
			ANNOTATIONS = 2

		filepath = self._filepath()

		with open(filepath, "r") as f:
			lines = f.readlines()
			section = Section.IMAGE_PATH
			object_count = 0
			object_counter = 0
			labels = []
			for line_no, line in enumerate(lines):
				line = line.rstrip()
				if section == Section.IMAGE_PATH:
					dirname = "WIDER_" + self.split
					img_path = os.path.join(self.root, dirname, "images", line)
					img_path = os.path.abspath(os.path.expanduser(img_path))
					section = Section.OBJECT_COUNT
				elif section == Section.OBJECT_COUNT:
					try:
						object_count = int(line)
						section = Section.ANNOTATIONS
					except ValueError:
						raise RuntimeError(
							"Error parsing WIDER Face annotation file " +
							filepath + ": " +
							"can't parse object count as integer at line " +
							line_no)

					num_boxes = int(line)
					num_boxes_line = False
					box_annotation_line = True
				elif section == Section.ANNOTATIONS:
					line_split = line.split(" ")
					line_values = [int(x) for x in line_split]
					labels.append(line_values)
					object_counter += 1
					if object_counter >= object_count:
						box_annotation_line = False
						file_name_line = True
						labels_tensor = torch.tensor(labels)
						self._data.append({
							"img_path": img_path,
							"annotations": {
								"bbox": labels_tensor[:, 0:4], # x, y, w, h
								"blur": labels_tensor[:, 4],
								"expression": labels_tensor[:, 5],
								"illumination": labels_tensor[:, 6],
								"occlusion": labels_tensor[:, 7],
								"pose": labels_tensor[:, 8],
								"invalid": labels_tensor[:, 9]}
						})
						object_counter = 0
						labels.clear()
						section = Section.IMAGE_PATH
				else:
					raise RuntimeError(
						"Error parsing WIDER Face annotation file " +
						filepath + " at line " + line_no)

	def _parse_test_annotations_file(self):
		filepath = self._filepath()
		with open(filepath, "r") as f:
			lines = f.readlines()
			for line_no, line in enumerate(lines):
				line = line.rstrip()
				dirname = "WIDER_" + self.split
				img_path = os.path.join(self.root, dirname, "images", line)
				img_path = os.path.abspath(os.path.expanduser(img_path))
				self._data.append({
					"img_path": img_path,
					"annotations": {}
				})

	def _read_image(self, img_path):
		image_file = self._get_abs_path(img_path)

		image = cv2.imread(image_file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		return image
