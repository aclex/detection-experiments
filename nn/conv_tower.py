import torch
from torch import nn


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
