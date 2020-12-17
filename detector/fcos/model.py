import torch

from torch import nn

from backbone.rw_mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small

from nn.separable_conv_2d import SeparableConv2d

from detector.model import Model
from detector.fcos.head import Head

from fpn.bifpn import BiFPN

class MobileNetV3SmallBiFPNFCOS(Model):
	def __init__(
			self, num_classes, num_channels=Head.DEFAULT_WIDTH,
			pretrained=False):
		backbone = MobileNetV3_Small(pretrained=pretrained)
		conv = SeparableConv2d
		num_levels = 5
		num_blocks = 2

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
