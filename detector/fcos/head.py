import math
import torch
from torch import nn

from .scale import Scale


class Tower(nn.Sequential):
	def __init__(
			self, num_channels, num_blocks,
			conv=nn.Conv2d, act=nn.ReLU, norm=nn.BatchNorm2d):
		self.num_blocks = num_blocks

		with_bias = bool(norm is None)

		nodes = []

		for i in range(self.num_blocks):
			node = nn.Sequential(
				conv(
					num_channels, num_channels,
					kernel_size=3, stride=1, padding=1,
					bias=with_bias),
				norm(num_features=num_channels) if with_bias else nn.Identity(),
				act())

			nodes.append(node)

		super(Tower, self).__init__(*nodes)


class Head(nn.Module):
	DEFAULT_NUM_BLOCKS = 4
	DEFAULT_WIDTH = 256
	DEFAULT_STRIDES = [8, 16, 32, 64, 128]

	def __init__(
			self, num_classes, num_channels=DEFAULT_WIDTH,
			conv=nn.Conv2d, act=nn.ReLU, norm=nn.BatchNorm2d,
			strides=DEFAULT_STRIDES, num_blocks=DEFAULT_NUM_BLOCKS):
		super(Head, self).__init__()

		self.num_blocks = num_blocks
		self.num_classes = num_classes
		self.num_levels = len(strides)

		with_bias = bool(norm is None)

		self.cls_tower = Tower(num_channels, num_blocks, conv, act, norm)
		self.reg_tower = Tower(num_channels, num_blocks, conv, act, norm)

		self.cls = nn.Conv2d(
			num_channels, num_classes,
			kernel_size=3, stride=1, padding=1)

		self.reg = nn.Conv2d(
			num_channels, 4,
			kernel_size=3, stride=1, padding=1)

		self.centerness = nn.Conv2d(
			num_channels, 1,
			kernel_size=3, stride=1, padding=1)

		self.scales = nn.ModuleList(
			[nn.Sequential(
				Scale(init_value=1.0),
				nn.ReLU()) for _ in range(self.num_levels)])

	def forward(self, x):
		assert len(x) == self.num_levels

		result = []

		for l, (xl, scale) in enumerate(zip(x, self.scales)):
			cls_tower_out = self.cls_tower.forward(xl)
			reg_tower_out = self.reg_tower.forward(xl)

			cls_out = self.cls.forward(cls_tower_out)
			reg_out = self.reg.forward(reg_tower_out)
			centerness_out = self.centerness.forward(reg_tower_out)

			scales_out = scale.forward(reg_out)

			joint_level_out = torch.cat([
				scales_out, centerness_out, cls_out], dim=1)

			result.append(joint_level_out)

		return result
