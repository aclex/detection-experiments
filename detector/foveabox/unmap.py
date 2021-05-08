import torch

from torch import nn

from detector.fcos import unmap as fcos_unmap


class Unmapper(fcos_unmap.Unmapper):
	def __init__(
			self, strides, image_size, num_classes,
			batch_size=1, device=None, dtype=torch.float32):
		super().__init__(
			strides, image_size, num_classes,
			batch_size, device, dtype)

	def _unmap_level(self, level, maps):
		s = self.strides[level]

		maps = maps.permute(0, 2, 3, 1)
		sp = self.split_joint_tensor(
			maps, self.num_classes, with_centerness=False)
		reg_level_map, cls_level_map = sp

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
