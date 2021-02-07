import torch

import math

from torch import nn

from .level_map_operations import LevelMapOperations


class Mapper(nn.Module, LevelMapOperations):
	def __init__(
			self, strides, image_size, num_classes,
			device=None, dtype=None):
		super(Mapper, self).__init__()

		self.strides = strides
		self.num_levels = len(self.strides)
		self.image_size = image_size
		self.num_classes = num_classes

		self.level_thresholds = self._calc_level_thresholds(strides, image_size)

		self._empty_maps = None
		self._grid_maps = None

		if device is not None or dtype is not None:
			self._empty_maps = self._create_empty_maps(device, dtype)
			self._grid_maps = self._create_grid_maps(device, dtype)

	def forward(self, x):
		"""Maps groud-truth targets to a bunch of per-level map tensors.

		Arguments:
			x: tuple(list(torch.Tensor)) - per-image labels and bounding boxes

		Returns:
			list(torch.Tensor) - per-level list of map tensors, i.e. imaginary
				shape is (L, B, C, H, W), where L - number of levels, B - batch
				size
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

			pixel_size = float(s) / image_size

			th_max = math.ceil(last_size / s)

			if th_max % 2:
				th_max += 1

			th_min = th_max // 2

			last_size = th_min * s

			if i == 0:
				th_min = 1

			result.append((th_min * pixel_size, th_max * pixel_size))

		return tuple(result[::-1])

	@staticmethod
	def _calc_level_map_sizes(strides, image_size):
		return [image_size // s for s in strides]

	@staticmethod
	def _create_level_map(stride, image_size, value=0., num_cell_elements=1):
		size = image_size // stride
		return torch.full((size, size, num_cell_elements), value)

	def _create_empty_maps(self, device, dtype):
		result = []

		for level in range(self.num_levels):
			s = self.strides[level]

			cls_level_map = self._create_level_map(
				s, self.image_size, num_cell_elements=self.num_classes)
			cls_level_map[..., 0] = 1. # set all pixels to 'background' class
			cls_level_map = cls_level_map.to(device=device, dtype=dtype)

			reg_level_map = self._create_level_map(
				s, self.image_size, num_cell_elements=4)
			reg_level_map = reg_level_map.to(device=device, dtype=dtype)

			centerness_level_map = self._create_level_map(s, self.image_size)
			centerness_level_map = centerness_level_map.to(
				device=device, dtype=dtype)

			result.append((reg_level_map, centerness_level_map, cls_level_map))

		return tuple(result)

	def _create_grid_maps(self, device, dtype):
		result = []

		for level in range(self.num_levels):
			s = self.strides[level]

			mx, my = self._create_level_reg_maps(s, self.image_size)
			mx = mx.to(device=device, dtype=dtype)
			my = my.to(device=device, dtype=dtype)

			result.append((mx, my))

		return tuple(result)

	@staticmethod
	def _calc_area(box):
		return (box[2] - box[0]) * (box[3] - box[1])

	@staticmethod
	def _calc_reg_slab(stride, subparts):
		result = torch.cat(subparts, dim=-1)

		return result

	@staticmethod
	def _calc_centerness_slab(l, t, r, b):
		return torch.sqrt(
			(torch.minimum(l, r) / torch.maximum(l, r)) *
			(torch.minimum(t, b) / torch.maximum(t, b))).to(l.device)

	def _calc_class_slab(self, stride, label):
		m = self._create_level_map(
			stride, self.image_size, num_cell_elements=self.num_classes)

		m[..., int(label)] = 1.

		m = m.to(label.device)

		return m

	@staticmethod
	def _filter_background(level_map):
		mn, _ = level_map.min(dim=2)
		return mn.unsqueeze(-1) >= 0

	def _pointwise_fit_in_level(self, level_map, level):
		th = self.level_thresholds[level]

		mx, _ = level_map.max(dim=2)

		return (mx.unsqueeze(-1) > th[0]) & (mx.unsqueeze(-1) <= th[1])

	def _mask(self, level_map, level):
		return self._filter_background(level_map) & \
			self._pointwise_fit_in_level(level_map, level)

	def _clear_box_background(self, cls_level_map, reg_level_map):
		fg = self._filter_background(reg_level_map)
		fg = fg.expand_as(cls_level_map).clone()
		fg[..., 1:] = False

		cls_level_map[fg] = 0.

	def _map_sample(self, gt_boxes, gt_labels):
		result = []

		gt_boxes *= self.image_size
		for level in range(self.num_levels):
			s = self.strides[level]

			if self._empty_maps is None:
				if len(gt_boxes) == 0:
					raise RuntimeError(
						"'device' and 'dtype' of level maps neither specified "
						"nor may be inferred due to no positive targets "
						"in the very first sample: either specify them "
						"explicitly or provide samples with at least one "
						"positive target")

				self._empty_maps = self._create_empty_maps(
					gt_boxes[0].device, gt_boxes[0].dtype)
				self._grid_maps = self._create_grid_maps(
					gt_boxes[0].device, gt_boxes[0].dtype)

			empty_maps = self._empty_maps[level]
			reg_empty_map, centerness_empty_map, cls_empty_map = empty_maps

			grid_maps = self._grid_maps[level]

			mx, my = grid_maps

			cls_level_map = cls_empty_map
			reg_level_map = reg_empty_map
			centerness_level_map = centerness_empty_map

			for box, label in sorted(
					zip(gt_boxes, gt_labels),
					key=lambda v: Mapper._calc_area(v[0]),
					reverse=True):
				norm_box = box / self.image_size

				l = mx - norm_box[0]
				t = my - norm_box[1]
				r = norm_box[2] - mx
				b = norm_box[3] - my

				reg_slab = self._calc_reg_slab(s, [l, t, r, b])
				centerness_slab = self._calc_centerness_slab(l, t, r, b)
				cls_slab = self._calc_class_slab(s, label)

				pred = self._mask(reg_slab, level)

				self._clear_box_background(cls_level_map, reg_slab)

				cls_level_map = torch.where(pred, cls_slab, cls_level_map)
				reg_level_map = torch.where(pred, reg_slab, reg_level_map)
				centerness_level_map = torch.where(
					pred, centerness_slab, centerness_level_map)

			level_map = torch.cat([
				reg_level_map,
				centerness_level_map,
				cls_level_map], dim=-1)

			level_map = level_map.permute(2, 0, 1)

			result.append(level_map)

		return result
