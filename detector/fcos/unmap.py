import torch

from torch import nn

from .level_map_operations import LevelMapOperations


class Unmapper(nn.Module, LevelMapOperations):
	def __init__(
			self, strides, image_size, num_classes,
			batch_size=1, device=None, dtype=torch.float32):
		super(Unmapper, self).__init__()

		self.strides = strides
		self.num_levels = len(self.strides)
		self.image_size = image_size
		self.batch_size = batch_size
		self.num_classes = num_classes

		self._diff_maps = self._create_diff_maps()

	def forward(self, x):
		"""Maps a bunch of per-level map tensors back to per-image targets.

		Arguments:
			x: list(torch.Tensor) - per-image labels and bounding boxes
		"""

		predictions = ([], [])

		for level, maps in enumerate(x):
			b, l = self._unmap_level(level, maps)

			predictions[0].append(b)
			predictions[1].append(l)

		reg = torch.cat(predictions[0], dim=1)
		cls = torch.cat(predictions[1], dim=1)

		return cls, reg

	@classmethod
	def _create_diff_map(cls, stride, image_size):
		mx, my = cls._create_level_reg_maps(stride, image_size)

		return torch.cat([mx, my, mx, my], dim=-1)

	def _create_diff_maps(self):
		result = []

		for level in range(self.num_levels):
			s = self.strides[level]

			m = self._create_diff_map(s, self.image_size)

			result.append(m)

		return tuple(result)

	def _unmap_reg_level(self, level, reg_level_map):
		reg_diff_map = self._diff_maps[level].to(reg_level_map)
		fix_sign = reg_level_map.new_tensor([-1, -1, 1, 1])

		return fix_sign * reg_level_map + reg_diff_map

	def _unmap_level(self, level, maps):
		s = self.strides[level]

		maps = maps.permute(0, 2, 3, 1)
		sp = self.split_joint_tensor(maps, self.num_classes)
		reg_level_map, centerness_level_map, cls_level_map = sp

		centerness_level_map = centerness_level_map.sigmoid()
		cls_level_map = cls_level_map.sigmoid()

		confidence_level_map = cls_level_map
		confidence_level_map = torch.max(
			confidence_level_map, torch.zeros_like(confidence_level_map))

		reg_level_map = self._unmap_reg_level(level, reg_level_map)
		reg_level_map = reg_level_map * self.image_size

		reg = reg_level_map.reshape(self.batch_size, -1, 4)
		reg /= self.image_size # convert to relative coordinates
		cls = confidence_level_map.reshape(
			self.batch_size, -1, self.num_classes)

		return reg, cls
