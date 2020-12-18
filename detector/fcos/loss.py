import torch

from functools import partial

from torch import nn
from torch.nn import functional as F

from torchvision.ops.focal_loss import sigmoid_focal_loss

from loss.ciou import box_ciou

from .level_map_operations import LevelMapOperations
from .unmap import Unmapper


class Loss(nn.Module):
	def __init__(self, strides, image_size, num_classes):
		super(Loss, self).__init__()

		self.reg_loss = box_ciou
		self.centerness_loss = partial(
			F.binary_cross_entropy_with_logits, reduction="sum")
		self.cls_loss = partial(sigmoid_focal_loss, reduction="sum")

		self.strides = strides
		self.image_size = image_size
		self.num_classes = num_classes

		self._unmapper = Unmapper(strides, image_size, num_classes)

	def forward(self, x, y):
		"""Calculates composite loss value

		Arguments:
		x: list(torch.Tensor) - per-level list of predicted map tensors,
				i.e. imaginary shape is (L, B, C, H, W), where L - number of
				levels, B - batch size
		y: list(torch.Tensor) - per-level list of target map tensors,
				i.e. imaginary shape is (L, B, C, H, W), where L - number of
				levels, B - batch size

		"""
		reg_pred, centerness_pred, cls_pred = self._flatten(x)
		reg_target, centerness_target, cls_target = self._flatten(y)

		centerness_sum = centerness_target.sum()

		mask = self._mask(reg_target, centerness_target, cls_target)

		num_elements = mask.numel()

		reg_pred, centerness_pred, cls_pred = Loss._apply_mask(
			reg_pred, centerness_pred, cls_pred, mask)
		reg_target, centerness_target, cls_target = Loss._apply_mask(
			reg_target, centerness_target, cls_target, mask)

		reg_loss = self.reg_loss(reg_pred, reg_target)
		reg_loss = torch.sum(reg_loss.flatten())
		if centerness_sum != 0:
			reg_loss /= centerness_sum

		centerness_loss = self.centerness_loss(
			centerness_pred, centerness_target)
		if num_elements != 0:
			centerness_loss /= num_elements

		cls_loss = self.cls_loss(cls_pred, cls_target)
		if num_elements != 0:
			cls_loss /= num_elements

		return {
			"total": self._combine(reg_loss, centerness_loss, cls_loss),
			"reg": reg_loss,
			"centerness": centerness_loss,
			"cls": cls_loss }

	@staticmethod
	def _combine(reg_loss, centerness_loss, cls_loss):
		return reg_loss + centerness_loss + cls_loss

	@staticmethod
	def _mask(reg_target, centerness_target, cls_target):
		mx, _ = torch.max(cls_target, dim=-1)
		return torch.nonzero(mx).flatten()

	@staticmethod
	def _apply_mask(reg, centerness, cls, mask):
		return reg[mask], centerness[mask], cls[mask]

	def _flatten(self, level_maps):
		reg_levels = []
		centerness_levels = []
		cls_levels = []

		for l, joint_map in enumerate(level_maps):
			s = self.strides[l]

			joint_map = joint_map.permute(0, 2, 3, 1)
			sp = LevelMapOperations.split_joint_tensor(
				joint_map, self.num_classes)
			reg_level_map, centerness_level_map, cls_level_map = sp

			# [l, r, t, b] -> [x1, y1, x2, y2]
			reg_level_map = self._unmapper._unmap_reg_level(l, reg_level_map)

			reg_levels.append(reg_level_map.reshape(-1, 4))
			centerness_levels.append(centerness_level_map.reshape(-1, 1))
			cls_levels.append(cls_level_map.reshape(-1, self.num_classes))

		reg_flat = torch.cat(reg_levels, dim=0)
		centerness_flat = torch.cat(centerness_levels, dim=0)
		cls_flat = torch.cat(cls_levels, dim=0)

		return reg_flat, centerness_flat, cls_flat
