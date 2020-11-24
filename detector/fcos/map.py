import torch

import math

from torch import nn

class Mapper(nn.Module):
	def __init__(self, strides, image_size, num_classes):
		super(Mapper, self).__init__()

		self.strides = strides
		self.num_levels = len(self.strides)
		self.image_size = image_size
		self.num_classes = num_classes

		self.level_thresholds = self._calc_level_thresholds(strides, image_size)

	def forward(self, x):
		return None

	@staticmethod
	def _calc_level_thresholds(strides, image_size):
		result = []

		last_size = image_size
		for i in range(len(strides) - 1, -1, -1):
			s = strides[i]

			th_max = math.ceil(last_size / s)
			th_min = max(th_max - 4, 1)

			assert th_max > th_min

			last_size = th_min * s

			result.append((th_min, th_max))

		return tuple(result[::-1])

	@staticmethod
	def _calc_level_map_sizes(strides, image_size):
		return [image_size // s for s in strides]

	@staticmethod
	def _create_level_reg_maps(stride, image_size):
		r = torch.arange(0, 1, float(stride) / image_size)

		my, mx = torch.meshgrid(r, r)

		return mx.unsqueeze(-1), my.unsqueeze(-1)

	@staticmethod
	def _create_level_map(stride, image_size, value=0., num_cell_elements=1):
		size = image_size // stride
		return torch.full((size, size, num_cell_elements), value)

	@staticmethod
	def _calc_area(box):
		return (box[2] - box[0]) * (box[3] - box[1])

	def _calc_class_slab(stride, label):
		m = self._create_level_map(
			stride, self.image_size, num_cell_elements=self.num_classses)

		m[..., label] = 1.

		return m

	@staticmethod
	def _calc_centerness_slab(l, t, r, b):
		return torch.sqrt(
			(torch.minimum(l, r) / torch.maximum(l, r)) *
			(torch.minimum(t, b) / torch.maximum(t, b)))

	@staticmethod
	def _filter_background(level_map):
		mn, _ = level_map.min(dim=2)
		return mn.unsqueeze(-1) >= 0

	def _pointwise_fit_in_level(self, level_map, level):
		th = self.level_thresholds[level]

		mn, _ = level_map.min(dim=2)
		mx, _ = level_map.max(dim=2)

		return (mn.unsqueeze(-1) > th[0]) & (mx.unsqueeze(-1) <= th[1])

	def _mask(self, level_map, level):
		return self._filter_background(level_map) & \
			self._pointwise_fit_in_level(level_map, level)

	def _map_sample(self, gt_boxes, gt_labels):
		level_maps = []
		for level in range(self.num_levels):
			s = self.strides[level]

			cls_level_map = self._create_level_map(
				s, self.image_size, num_cell_elements=self.num_classes)
			reg_level_map = self._create_level_map(
				s, self.image_size, value=float("+inf"))
			centerness_level_map = self._create_level_map(s, self.image_size)

			for box, label in sorted(
					zip(gt_boxes, gt_labels),
					key=lambda v: Mapper._calc_area(v[0]),
					reverse=True):
				mx, my = self._create_level_reg_maps(s, self.image_size)

				l = mx - math.ceil(float(box[0]) / s)
				t = my - math.ceil(float(box[1]) / s)
				r = math.floor(float(box[2]) / s) - mx
				b = math.floor(float(box[3]) / s) - my

				cls_slab = self._create_level_map(
					s, self.image_size, num_cell_elements=self.num_classes)
				reg_slab = torch.cat([l, t, r, b], dim=-1)
				centerness_slab = self._calc_centerness_slab(l, t, r, b)

				pred = self._mask(reg_slab, level)

				cls_level_map = torch.where(pred, cls_slab, cls_level_map)
				reg_level_map = torch.where(pred, reg_slab, reg_level_map)
				centerness_level_map = torch.where(
					pred, centerness_slab, centerness_level_map)

			level_map = torch.cat([
				reg_level_map,
				centerness_level_map,
				cls_level_map], dim=-1)

			level_maps.append(level_map)

		return level_maps
