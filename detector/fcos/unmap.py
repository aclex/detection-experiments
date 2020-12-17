import torch

from torch import nn

from .level_map_operations import LevelMapOperations


class Unmapper(nn.Module, LevelMapOperations):
	def __init__(
			self, strides, image_size, num_classes,
			batch_size=1, prefilter_threshold=0.,
			device=None, dtype=None):
		super(Unmapper, self).__init__()

		self.strides = strides
		self.num_levels = len(self.strides)
		self.image_size = image_size
		self.batch_size = batch_size
		self.num_classes = num_classes

		self.prefilter_threshold = prefilter_threshold

		self._diff_maps = None
		self._fix_sign = None

		if device is not None or dtype is not None:
			self._diff_maps = self._create_diff_maps(device, dtype)
			self._fix_sign = self._create_fix_sign(device, dtype)

	def forward(self, x):
		"""Maps a bunch of per-level map tensors back to per-image targets.

		Arguments:
			x: list(torch.Tensor) - per-image labels and bounding boxes
		"""

		predictions = ([], [])
		for i in range(self.batch_size):
			image_predictions = ([], [])

			for level, maps in enumerate(x):
				b, l = self._unmap_level(level, maps)

				image_predictions[0].append(b)
				image_predictions[1].append(l)

			predictions[0].append(torch.cat(image_predictions[0], dim=0))
			predictions[1].append(torch.cat(image_predictions[1], dim=0))

		return predictions

	@classmethod
	def _create_diff_map(cls, stride, image_size):
		mx, my = cls._create_level_reg_maps(stride, image_size)

		return torch.cat([mx, my, mx, my], dim=-1)

	def _create_diff_maps(self, device, dtype):
		result = []

		for level in range(self.num_levels):
			s = self.strides[level]

			m = self._create_diff_map(s, self.image_size)
			m = m.to(device=device, dtype=dtype)

			result.append(m)

		return tuple(result)

	@staticmethod
	def _create_fix_sign(device, dtype):
		return torch.tensor([-1, -1, 1, 1]).to(device=device, dtype=dtype)

	def _unmap_reg_level(self, level, reg_level_map):
		if self._diff_maps is None or self._fix_sign is None:
			self._diff_maps = self._create_diff_maps(
				device=reg_level_map.device, dtype=reg_level_map.dtype)
			self._fix_sign = self._create_fix_sign(
				device=reg_level_map.device, dtype=reg_level_map.dtype)

		reg_diff_map = self._diff_maps[level]

		return self._fix_sign * reg_level_map + reg_diff_map

	@staticmethod
	def _create_prefilter_mask(centered_cls_level_map, threshold):
		mx, _ = centered_cls_level_map.max(dim=-1)
		return mx >= threshold

	def _unmap_level(self, level, maps):
		s = self.strides[level]

		maps = maps.permute(1, 2, 0)
		sp = self.split_joint_tensor(maps, self.num_classes)
		reg_level_map, centerness_level_map, cls_level_map = sp

		reg_level_map *= s

		centered_cls_level_map = centerness_level_map * cls_level_map

		mask = self._create_prefilter_mask(
			centered_cls_level_map, self.prefilter_threshold)

		reg_level_map = self._unmap_reg_level(level, reg_level_map)

		reg_level_map = reg_level_map[mask]
		centered_cls_level_map = centered_cls_level_map[mask]

		reg = reg_level_map.reshape(-1, 4)
		cls = centered_cls_level_map.reshape(-1, self.num_classes)

		return reg, cls
