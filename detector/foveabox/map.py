import math

import torch

from torch import nn

from detector.fcos import map as fcos_map


class Mapper(fcos_map.Mapper):
	def __init__(
			self, strides, image_size, num_classes,
			sigma=1, atss_k=None,
			device=None, dtype=None):
		super().__init__(strides, image_size, num_classes, device, dtype)

		self.sigma = sigma if atss_k is None else 1
		self.atss_k = atss_k

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

			th_min = 1

			last_size = (th_max // 2) * s

			result.append((th_min * pixel_size, th_max * pixel_size))

		return tuple(result[::-1])

	def _is_visible(self, box, level):
		s = self.strides[level]

		return self.sigma * (box[2] - box[0]) >= s and \
			self.sigma * (box[3] - box[1]) >= s

	def _fits_in_level(self, area, level):
		norm_area = area / (self.image_size * self.image_size)
		th = self.level_thresholds[level]

		return (norm_area > th[0] * th[0]) & (norm_area <= th[1] * th[1])

	def _calc_atss_slab(self, level_map):
		atss_map = level_map.clone().detach()

		l = atss_map[..., 0]
		t = atss_map[..., 1]
		r = atss_map[..., 2]
		b = atss_map[..., 3]
		fg = self._filter_background(atss_map).squeeze(-1)

		k = min(torch.nonzero(fg).numel(), self.atss_k)

		atss_map = torch.abs(l - r) + torch.abs(t - b)
		neutral = atss_map.new_full(size=atss_map.shape, fill_value=2)
		atss_map = torch.where(fg > 0, atss_map, neutral)
		flatten_atss_map = atss_map.flatten()

		_, indices = flatten_atss_map.topk(k, largest=False)

		z = torch.zeros_like(flatten_atss_map)
		result = z.scatter(-1, indices, 2).reshape_as(atss_map)
		result -= 1
		result.unsqueeze_(-1)

		return result

	def _calc_fovea_slab(self, box, level_map):
		width = box[2] - box[0]
		height = box[3] - box[1]

		result = level_map.clone().detach()
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
				norm_box = box / self.image_size

				l = mx - norm_box[0]
				t = my - norm_box[1]
				r = norm_box[2] - mx
				b = norm_box[3] - my

				reg_slab = self._calc_reg_slab(s, [l, t, r, b])
				cls_slab = self._calc_class_slab(s, label)

				self._clear_box_background(cls_level_map, reg_slab)

				if not self._is_visible(box, level) or \
						not self._fits_in_level(area, level):
					continue

				if self.atss_k is not None:
					fovea_slab = self._calc_atss_slab(reg_slab)
				else:
					fovea_slab = self._calc_fovea_slab(norm_box, reg_slab)

				reg_pred = self._filter_background(reg_slab)
				cls_pred = self._filter_background(fovea_slab)

				cls_level_map = torch.where(cls_pred, cls_slab, cls_level_map)
				reg_level_map = torch.where(reg_pred, reg_slab, reg_level_map)

			level_map = torch.cat([reg_level_map, cls_level_map], dim=-1)

			level_map = level_map.permute(2, 0, 1)

			result.append(level_map)

		return result
