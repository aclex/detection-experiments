import torch

from torch import nn

from backbone.rw_mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small

from nn.separable_conv_2d import SeparableConv2d

from detector.model import Model
from detector.fcos.fcos import Head

from fpn.bifpn import BiFPN

class MobileNetV3SmallBiFPNFCOS(Model):
	def __init__(self, num_classes, pretrained=False):
		backbone = MobileNetV3_Small(pretrained=pretrained)

		def fpn_builder(
				feature_channels, feature_strides, out_channels,
				conv, norm, act):
			return BiFPN(
				feature_channels=feature_channels,
				feature_strides=feature_strides,
				out_channels=out_channels,
				num_layers=1)

		head = Head(num_classes=num_classes, conv=SeparableConv2d,
			act=nn.Hardswish)

		super(MobileNetV3SmallBiFPNFCOS, self).__init__(
			backbone, fpn_builder, head, num_levels=5,
			fpn_channels=Head.DEFAULT_WIDTH)
