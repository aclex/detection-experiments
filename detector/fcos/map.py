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
		"""Maps groud-truth targets to a bunch of per-level map tensors.

		Arguments:
			x: tuple(list(torch.Tensor)) - per-image labels and bounding boxes
		"""
		batch_results = []

		for b, l in zip(*x):
			level_maps = self._map_sample(b, l)
			batch_results.append(level_maps)

		result = []
		for i in range(self.num_levels):
			per_level_maps = [t[i] for t in batch_results]
			r = torch.stack(per_level_maps)
			result.append(r)

		return result

	@staticmethod
	def _calc_level_thresholds(strides, image_size):
		result = []

		last_size = image_size
		for i in range(len(strides) - 1, -1, -1):
			s = strides[i]

			th_max = math.ceil(last_size / s)

			if th_max % 2:
				th_max += 1

			th_min = th_max / 2

			last_size = th_min * s

			result.append((th_min, th_max))

		return tuple(result[::-1])

	@staticmethod
	def _calc_level_map_sizes(strides, image_size):
		return [image_size // s for s in strides]

	@staticmethod
	def _create_level_reg_maps(stride, image_size):
		r = torch.arange(0, image_size, stride)

		my, mx = torch.meshgrid(r, r)

		return mx.unsqueeze(-1), my.unsqueeze(-1)

	@staticmethod
	def _create_level_map(stride, image_size, value=0., num_cell_elements=1):
		size = image_size // stride
		return torch.full((size, size, num_cell_elements), value)

	@staticmethod
	def _calc_area(box):
		return (box[2] - box[0]) * (box[3] - box[1])

	@staticmethod
	def _calc_reg_slab(stride, subparts):
		result = torch.cat(subparts, dim=-1)
		result /= stride

		return result

	@staticmethod
	def _calc_centerness_slab(l, t, r, b):
		return torch.sqrt(
			(torch.minimum(l, r) / torch.maximum(l, r)) *
			(torch.minimum(t, b) / torch.maximum(t, b)))

	def _calc_class_slab(self, stride, label):
		m = self._create_level_map(
			stride, self.image_size, num_cell_elements=self.num_classes)

		m[..., int(label)] = 1.

		return m

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
				s, self.image_size, num_cell_elements=4)
			centerness_level_map = self._create_level_map(s, self.image_size)

			for box, label in sorted(
					zip(gt_boxes, gt_labels),
					key=lambda v: Mapper._calc_area(v[0]),
					reverse=True):
				mx, my = self._create_level_reg_maps(s, self.image_size)

				l = mx - box[0]
				t = my - box[1]
				r = box[2] - mx
				b = box[3] - my

				reg_slab = self._calc_reg_slab(s, [l, t, r, b])
				centerness_slab = self._calc_centerness_slab(l, t, r, b)
				cls_slab = self._calc_class_slab(s, label)

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
