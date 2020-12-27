import torch

from torch import nn

from .level_map_operations import LevelMapOperations


class Unmapper(nn.Module, LevelMapOperations):
	def __init__(
			self, strides, image_size, num_classes,
			batch_size=1, device=None, dtype=None):
		super(Unmapper, self).__init__()

		self.strides = strides
		self.num_levels = len(self.strides)
		self.image_size = image_size
		self.batch_size = batch_size
		self.num_classes = num_classes

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

	def _unmap_level(self, level, maps):
		s = self.strides[level]

		level_map_size = self.image_size // s

		maps = maps.permute(0, 2, 3, 1)
		sp = self.split_joint_tensor(maps, self.num_classes)
		reg_level_map, centerness_level_map, cls_level_map = sp

		reg_level_map = reg_level_map * s

		centered_cls_level_map = centerness_level_map * cls_level_map
		centered_cls_level_map = torch.max(
			centered_cls_level_map, torch.zeros_like(centered_cls_level_map))

		reg_level_map = self._unmap_reg_level(level, reg_level_map)

		help_onnx_infer_inner_dim = level_map_size ** 2
		reg = reg_level_map.reshape(
			self.batch_size, help_onnx_infer_inner_dim, 4)
		reg /= self.image_size # convert to relative coordinates
		cls = centered_cls_level_map.reshape(
			self.batch_size, help_onnx_infer_inner_dim, self.num_classes)

		return reg, cls
