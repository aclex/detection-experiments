import math

import torch

from torch import nn

from detector.fcos import map as fcos_map


class Mapper(fcos_map.Mapper):
	def __init__(
			self, strides, image_size, num_classes,
			sigma,
			device=None, dtype=None):
		super().__init__(strides, image_size, num_classes, device, dtype)

		self.sigma = sigma

	def _is_visible(self, box, level):
		s = self.strides[level]

		return self.sigma * (box[2] - box[0]) >= s and \
			self.sigma * (box[3] - box[1]) >= s

	def _fits_in_level(self, area, level):
		norm_area = area / (self.image_size * self.image_size)
		th = self.level_thresholds[level]

		return (norm_area > th[0] * th[0]) & (norm_area <= th[1] * th[1])

	def _calc_fovea_slab(self, box, level_map):
		width = box[2] - box[0]
		height = box[3] - box[1]

		result = level_map.detach()
		result[..., (0, 2)] -= (1 - self.sigma) / 2. * width
		result[..., (1, 3)] -= (1 - self.sigma) / 2. * height

		return result

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
					gt_boxes[0].device, gt_boxes[0].dtype,
					with_centerness=False)
				self._grid_maps = self._create_grid_maps(
					gt_boxes[0].device, gt_boxes[0].dtype)

			empty_maps = self._empty_maps[level]
			reg_empty_map, cls_empty_map = empty_maps

			grid_maps = self._grid_maps[level]

			mx, my = grid_maps

			cls_level_map = cls_empty_map
			reg_level_map = reg_empty_map

			for box, label, area in sorted(
					zip(gt_boxes, gt_labels, map(Mapper._calc_area, gt_boxes)),
					key=lambda v: v[2],
					reverse=True):
				if not self._is_visible(box, level) or \
						not self._fits_in_level(area, level):
					continue

				norm_box = box / self.image_size

				l = mx - norm_box[0]
				t = my - norm_box[1]
				r = norm_box[2] - mx
				b = norm_box[3] - my

				reg_slab = self._calc_reg_slab(s, [l, t, r, b])
				fovea_slab = self._calc_fovea_slab(norm_box, reg_slab)
				cls_slab = self._calc_class_slab(s, label)

				reg_pred = self._filter_background(reg_slab)
				cls_pred = self._filter_background(fovea_slab)

				self._clear_box_background(cls_level_map, reg_slab)

				cls_level_map = torch.where(cls_pred, cls_slab, cls_level_map)
				reg_level_map = torch.where(reg_pred, reg_slab, reg_level_map)

			level_map = torch.cat([reg_level_map, cls_level_map], dim=-1)

			level_map = level_map.permute(2, 0, 1)

			result.append(level_map)

		return result
