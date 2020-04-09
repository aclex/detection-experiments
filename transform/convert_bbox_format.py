import albumentations as A

from albumentations.augmentations.bbox_utils import (
	convert_bbox_from_albumentations,
	convert_bbox_to_albumentations,
	normalize_bbox,
	denormalize_bbox,
	check_bbox
)


class ConvertBboxFormat(A.DualTransform):
	def __init__(self, source_format, target_format, rows, cols,
				 check_validity=False, always_apply=True, p=1.0):
		super(ConvertBboxFormat, self).__init__(always_apply, p)

		if source_format not in { "pascal_voc", "coco", "albumentations", "yolo" }:
			raise ValueError(
				"Unknown source_format {}. Supported formats are: 'coco', 'pascal_voc', 'albumentations' and 'yolo'".format(target_format)
			)

		if target_format not in { "pascal_voc", "coco", "albumentations", "yolo" }:
			raise ValueError(
				"Unknown target_format {}. Supported formats are: 'coco', 'pascal_voc', 'albumentations' and 'yolo'".format(target_format)
			)

		self.source_format = source_format
		self.target_format = target_format
		self.rows = rows
		self.cols = cols
		self.check_validity = check_validity

	@property
	def targets(self):
		super_targets = super(ConvertBboxFormat, self).targets

		return { t: p  for t, p in super_targets.items() if t == "bboxes" }

	def apply_to_bbox(self, bbox, **params):
		if self.source_format == 'albumentations':
			return convert_bbox_from_albumentations(bbox, self.target_format,
													self.rows, self.cols,
													self.check_validity)

		elif self.target_format == 'albumentations':
			return convert_bbox_to_albumentations(bbox, self.source_format,
												  self.rows, self.cols,
												  self.check_validity)

		else:
			if self.check_validity:
				check_bbox(bbox)

			if self.target_format == 'pascal_voc':
				return self._convert_to_pascal_voc(bbox)

			elif self.target_format == 'coco':
				return self._convert_to_coco(bbox)

			elif self.target_format == 'yolo':
				return self._convert_to_yolo(bbox)

	def _convert_to_pascal_voc(self, bbox):
		(x, y, width, height), tail = bbox[:4], bbox[4:]

		if self.source_format == 'coco':
			return (x, y, x + width, y + height) + tail

		elif self.source_format == 'yolo':
			_bbox = np.array(bbox[:4])
			if np.any((_bbox <= 0) | (_bbox > 1)):
				raise ValueError("In YOLO format all labels must be float and in range (0, 1]")

			x, y, width, height = np.round(denormalize_bbox(bbox, rows, cols))

			x_min = x - width / 2 + 1
			x_max = x_min + width
			y_min = y - height / 2 + 1
			y_max = y_min + height

			return (x_min, y_min, x_max, y_max) + tail

	def _convert_to_coco(self, bbox):
		if self.source_format == 'pascal_voc':
			(x1, y1, x2, y2), tail = bbox[:4], bbox[4:]
			return (x1, y1, x2 - x1, y2 - y1) + tail

		elif self.source_format == 'yolo':
			_bbox = np.array(bbox[:4])
			if np.any((_bbox <= 0) | (_bbox > 1)):
				raise ValueError("In YOLO format all labels must be float and in range (0, 1]")

			x, y, width, height = np.round(denormalize_bbox(bbox, rows, cols))

			x_min = x - width / 2 + 1
			y_min = y - height / 2 + 1

			return (x_min, y_min, width, height) + tail

	def _convert_to_yolo(self, bbox):
		if self.source_format == 'pascal_voc':
			(x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]
			return normalize_bbox((
				(x_max - x_min) / 2, (y_max - y_min) / 2,
				(x_max - x_min), (y_max - y_min)
			) + tail, self.rows, self.cols)

		elif self.source_format == 'coco':
			(x, y, width, height), tail = bbox[:4], bbox[4:]
			return normalize_bbox((
				x + width / 2,
				y + height / 2,
				width,
				height
			) + tail, self.rows, self.cols)
