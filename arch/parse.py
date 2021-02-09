from functools import partial
from torch import nn

from backbone.rw_mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from backbone.ghostnet import GhostNet

from fpn.bifpn import BiFPN

from nn.separable_conv_2d import SeparableConv2d


def parse_backbone(settings):
	arch = settings["arch"]
	if arch.casefold() == "mobilenetv3-small":
		return MobileNetV3_Small

	elif arch.casefold() == "mobilenetv3-large":
		return MobileNetV3_Large

	elif arch.casefold() == "ghostnet":
		width = settings.get("width", 1.0)
		return partial(GhostNet, width=width)

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
	if name is None or name.casefold() == "relu":
		return nn.ReLU
	elif name.casefold() == "relu6":
		return nn.ReLU6
	elif name.casefold() == "leakyrelu":
		return nn.LeakyReLU
