import torch

from torch import nn

from .level_map_operations import LevelMapOperations


class Unmapper(nn.Module, LevelMapOperations):
	def __init__(
			self, strides, image_size, batch_size, num_classes,
			prefilter_threshold=0.):
		super(Unmapper, self).__init__()

		self.strides = strides
		self.num_levels = len(self.strides)
		self.image_size = image_size
		self.batch_size = batch_size
		self.num_classes = num_classes

		self.prefilter_threshold = prefilter_threshold

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

	@staticmethod
	def _fix_sign():
		return torch.tensor([-1, -1, 1, 1], dtype=torch.float32)

	@staticmethod
	def _create_prefilter_mask(centered_cls_level_map, threshold):
		mx, _ = centered_cls_level_map.max(dim=-1)
		return mx >= threshold

	def _unmap_level(self, level, maps):
		s = self.strides[level]

		sp = torch.split(maps, [4, 1, self.num_classes], dim=-1)
		reg_level_map, centerness_level_map, cls_level_map = sp

		reg_level_map *= s

		centered_cls_level_map = centerness_level_map * cls_level_map

		mask = self._create_prefilter_mask(
			centered_cls_level_map, self.prefilter_threshold)

		reg_diff_map = self._create_diff_map(s, self.image_size)

		sn = self._fix_sign()
		reg_level_map = sn * reg_level_map + reg_diff_map

		reg_level_map = reg_level_map[mask]
		centered_cls_level_map = centered_cls_level_map[mask]

		reg = reg_level_map.reshape(-1, 4)
		cls = centered_cls_level_map.reshape(-1, self.num_classes)

		return reg, cls
