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

			result.append((reg_level_map, cls_level_map))

		return tuple(result)

	def _is_visible(self, box, level):
		s = self.strides[level]

		return self.sigma * (box[2] - box[0]) >= s and \
			self.sigma * (box[3] - box[1]) >= s

	def _fits_in_level(self, area, level):
		norm_area = area / (self.image_size * self.image_size)
		th = self.level_thresholds[level]

		return (norm_area > th[0] * th[0]) & (norm_area <= th[1] * th[1])

	def _fovea(self, box, level):
		s = self.strides[level]
		mx, my = self._grid_maps[level]

		norm_box = box / self.image_size

		norm_width = norm_box[2] - norm_box[0]
		norm_height = norm_box[3] - norm_box[1]

		l = mx - (norm_box[0] + (1 - self.sigma) / 2. * norm_width)
		t = my - (norm_box[1] + (1 - self.sigma) / 2. * norm_height)
		r = norm_box[2] - (1 - self.sigma) / 2. * norm_width - mx
		b = norm_box[3] - (1 - self.sigma) / 2. * norm_height - my

		fovea_slab = self._calc_reg_slab(s, [l, t, r, b])
		mn, _ = fovea_slab.min(dim=2)

		return mn.unsqueeze(-1) >= 0

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
				cls_slab = self._calc_class_slab(s, label)

				reg_pred = self._filter_background(reg_slab)
				fovea = self._fovea(box, level)

				self._clear_box_background(cls_level_map, reg_slab)

				cls_level_map = torch.where(fovea, cls_slab, cls_level_map)
				reg_level_map = torch.where(reg_pred, reg_slab, reg_level_map)

			level_map = torch.cat([reg_level_map, cls_level_map], dim=-1)

			level_map = level_map.permute(2, 0, 1)

			result.append(level_map)

		return result
