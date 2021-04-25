from functools import partial
from torch import nn

from backbone.ghostnet import GhostNet
from backbone.tinynet import TinyNet
from backbone.timm import Timm

from fpn.bifpn import BiFPN

from nn.separable_conv_2d import SeparableConv2d
from nn.mish import Mish


def parse_backbone(settings):
	arch = settings["arch"]
	if arch.casefold() == "ghostnet":
		width = settings.get("width", 1.0)

		return partial(GhostNet, width=width)

	elif arch.casefold() == "tinynet":
		edition = settings.get("edition", None)
		r = settings.get("r", 1)
		w = settings.get("w", 1)
		d = settings.get("d", 1)

		return partial(TinyNet, edition=edition, r=r, w=w, d=d)

	else: # try to create model with `timm`
		return partial(Timm, model=arch)


def parse_fpn_class(name):
	if name.casefold() == "bifpn":
		return BiFPN

	else:
		return None


def parse_conv(name):
	if name is None or name.casefold() == "conv":
		return nn.Conv2d

	elif name.casefold() == "depthwise-separable" or \
			name.casefold() == "depthwise" or \
			name.casefold() == "separable":
		return SeparableConv2d


def parse_norm(name):
	if name is None or name.casefold() == "batch" or \
			name.casefold() == "batchnorm":
		return nn.BatchNorm2d

	if name.casefold() == "group" or name.casefold() == "groupnorm":
		return nn.GroupNorm


def parse_act(name):
	if name is None or name.casefold() == "relu":
		return nn.ReLU
	elif name.casefold() == "relu6":
		return nn.ReLU6
	elif name.casefold() == "leakyrelu":
		return nn.LeakyReLU
	elif name.casefold() == "gelu":
		return nn.GELU
	elif name.casefold() == "hardswish" or name.casefold() == "hswish":
		return nn.Hardswish
	elif name.casefold() == "mish":
		return Mish
