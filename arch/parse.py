from torch import nn

from backbone.rw_mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from backbone.ghostnet import GhostNet, GhostNet075, GhostNet050, GhostNet025

from fpn.bifpn import BiFPN

from nn.separable_conv_2d import SeparableConv2d


def parse_backbone(name):
	if name == "mobilenetv3-small":
		return MobileNetV3_Small

	elif name == "mobilenetv3-large":
		return MobileNetV3_Large

	elif name == "ghostnet-1.0":
		return GhostNet

	elif name == "ghostnet-0.75":
		return GhostNet075

	elif name == "ghostnet-0.50":
		return GhostNet050

	elif name == "ghostnet-0.25":
		return GhostNet025

	else:
		return None


def parse_fpn_class(name):
	if name == "bifpn":
		return BiFPN

	else:
		return None


def parse_conv(name):
	if name == "depthwise-separable" or \
			name == "depthwise" or \
			name == "separable":
		return SeparableConv2d

	else:
		return nn.Conv2d


def parse_norm(name):
	if name == "group":
		return nn.GroupNorm2d

	else:
		return nn.BatchNorm2d


def parse_act(name):
		return nn.ReLU
