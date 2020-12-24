import torch

from torch import nn

from detector.model import Model

from detector.fcos.head import Head
from detector.fcos.unmap import Unmapper

from fpn.bifpn import BiFPN


class Blueprint(Model):
	def __init__(
			self, arch_name, backbone, num_classes,
			num_channels=Head.DEFAULT_WIDTH,
			num_levels=Head.DEFAULT_NUM_LEVELS,
			num_fpn_layers=None,
			num_blocks=Head.DEFAULT_NUM_BLOCKS,
			fpn=BiFPN, conv=nn.Conv2d, norm=nn.BatchNorm2d, act=nn.ReLU):

		self.arch_name = arch_name

		def fpn_builder(
				feature_channels, feature_strides, out_channels,
				conv, norm, act):
			return fpn(
				feature_channels=feature_channels,
				feature_strides=feature_strides,
				out_channels=out_channels,
				num_layers=num_fpn_layers or 1,
				conv=conv, norm=norm, act=act)

		def head_builder(strides):
			return Head(
				num_classes=num_classes, strides=strides,
				conv=conv, act=act,
				num_channels=num_channels, num_blocks=num_blocks)

		super(Blueprint, self).__init__(
			backbone, fpn_builder, head_builder, num_levels=num_levels,
			fpn_channels=num_channels, conv=conv)


class BlueprintInference(Blueprint):
	def __init__(
			self, arch_name, backbone, num_classes, batch_size=None,
			num_channels=Head.DEFAULT_WIDTH,
			num_levels=Head.DEFAULT_NUM_LEVELS,
			num_fpn_layers=None,
			num_blocks=Head.DEFAULT_NUM_BLOCKS,
			fpn=BiFPN, conv=nn.Conv2d, norm=nn.BatchNorm2d, act=nn.ReLU):
		super(BlueprintInference, self).__init__(
			arch_name, backbone, num_classes,
			num_channels, num_levels, num_fpn_layers, num_blocks,
			fpn, conv, norm, act)

		self.batch_size = batch_size
		self._unmapper = None

	def forward(self, x):
		output = super(BlueprintInference, self).forward(x)

		if self._unmapper is None:
			self._unmapper = Unmapper(
				strides=self.strides, image_size=x.size(-1),
				num_classes=self.head.num_classes,
				batch_size=self.batch_size or x.size(0),
				device=x.device, dtype=x.dtype)

		return self._unmapper.forward(output)
