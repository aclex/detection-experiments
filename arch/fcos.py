import os
import json

import arch.default_settings

from detector.fcos.model import Blueprint, BlueprintInference
from detector.fcos.map import Mapper
from detector.fcos.loss import Loss

from arch.core_settings import CoreSettings
from arch.compound_scaling import CompoundScaling

from arch.parse import (
	parse_backbone,
	parse_fpn_class,
	parse_conv,
	parse_norm,
	parse_act)


class FCOS(CoreSettings):
	def __init__(self, config):
		with open(config, 'r') as f:
			self.settings = json.load(f)

		self.name = os.path.splitext(os.path.basename(config))[0]

		super(FCOS, self).__init__()

	def build(self, num_classes, pretrained_backbone=False):
		backbone_class = parse_backbone(self.settings["backbone"])
		backbone = backbone_class(pretrained=pretrained_backbone)

		fpn = parse_fpn_class(self.settings["fpn"])

		conv = parse_conv(self.settings.get("conv", None))
		norm = parse_norm(self.settings.get("norm", None))
		act = parse_act(self.settings.get("act", None))

		pheta = CompoundScaling.pheta(self.image_size)

		return Blueprint(
			self.name, backbone, num_classes,
			num_channels=CompoundScaling.fpn_width(pheta),
			num_levels=CompoundScaling.fpn_height(pheta),
			num_fpn_layers=CompoundScaling.fpn_depth(pheta),
			num_blocks=CompoundScaling.head_depth(pheta),
			fpn=fpn, conv=conv, norm=norm, act=act)

	def loss(self, net, device=None):
		return Loss(net.strides, self.image_size, net.head.num_classes)

	def mapper(self, net, device=None):
		return Mapper(net.strides, self.image_size, net.head.num_classes)
