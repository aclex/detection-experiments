import torch

from torch import nn

from backbone.rw_mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small

from nn.separable_conv_2d import SeparableConv2d

from detector.model import Model

from detector.fcos.head import Head
from detector.fcos.unmap import Unmapper

from fpn.bifpn import BiFPN

class MobileNetV3SmallBiFPNFCOS(Model):
	def __init__(
			self, num_classes, num_channels=Head.DEFAULT_WIDTH,
			pretrained=False):
		backbone = MobileNetV3_Small(pretrained=pretrained)
		conv = SeparableConv2d
		num_levels = 3
		num_blocks = 2

		self.arch_name = "mb3-small-bifpn-fcos"

		def fpn_builder(
				feature_channels, feature_strides, out_channels,
				conv, norm, act):
			return BiFPN(
				feature_channels=feature_channels,
				feature_strides=feature_strides,
				out_channels=out_channels,
				num_layers=1,
				conv=conv, norm=norm, act=act)

		def head_builder(strides):
			return Head(
				num_classes=num_classes, strides=strides,
				conv=conv, act=nn.ReLU,
				num_channels=num_channels, num_blocks=num_blocks)

		super(MobileNetV3SmallBiFPNFCOS, self).__init__(
			backbone, fpn_builder, head_builder, num_levels=num_levels,
			fpn_channels=num_channels, conv=conv)


class MobileNetV3SmallBiFPNFCOSInference(MobileNetV3SmallBiFPNFCOS):
	def __init__(
			self, num_classes, batch_size=None,
			num_channels=Head.DEFAULT_WIDTH):
		super(MobileNetV3SmallBiFPNFCOSInference, self).__init__(
			num_classes, num_channels)

		self.batch_size = batch_size
		self._unmapper = None


	def forward(self, x):
		output = super(MobileNetV3SmallBiFPNFCOSInference).forward(x)

		if self._unmapper is None:
			self._unmapper = Unmapper(
				strides=self.strides, image_size=x.size(-1),
				num_classes=num_classes,
				batch_size=self.batch_size or x.size(0),
				device=x.device, dtype=x.dtype)

		return self._unmapper.forward(output)
