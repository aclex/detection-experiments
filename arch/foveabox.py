import os
import json

from detector.foveabox.model import Blueprint, BlueprintInference
from detector.foveabox.map import Mapper
from detector.foveabox.loss import Loss

from arch.core_settings import CoreSettings
from arch.compound_scaling import CompoundScaling

from arch.parse import (
	parse_backbone,
	parse_fpn_class,
	parse_conv,
	parse_norm,
	parse_act)


class FoveaBox(CoreSettings):
	def __init__(self, config):
		super().__init__(config)

		self.backbone_class = parse_backbone(self.settings["backbone"])

		self.fpn = parse_fpn_class(self.settings["fpn"])

		self.conv = parse_conv(self.settings.get("conv", None))
		self.norm = parse_norm(self.settings.get("norm", None))
		self.act = parse_act(self.settings.get("act", None))

		self.pheta = CompoundScaling.pheta(self.image_size)
		self.sigma = self.settings.get("sigma", 0.4)
		self.atss_k_ratio = self.settings.get("atss", {}).get("k_ratio", None)

	def build(
			self, num_classes, pretrained_backbone=False,
			batch_size=1, inference=False):
		backbone = self.backbone_class(pretrained=pretrained_backbone)

		ctor = BlueprintInference if inference else Blueprint

		return ctor(
			self.name, backbone, num_classes,
			num_channels=CompoundScaling.fpn_width(self.pheta),
			num_levels=CompoundScaling.fpn_height(self.pheta),
			num_fpn_layers=CompoundScaling.fpn_depth(self.pheta),
			num_blocks=CompoundScaling.head_depth(self.pheta),
			fpn=self.fpn, conv=self.conv, norm=self.norm, act=self.act)

	def loss(self, net, device=None):
		return Loss(net.strides, self.image_size, net.head.num_classes)

	def mapper(self, net, device=None):
		return Mapper(
			net.strides, self.image_size, net.head.num_classes,
			sigma=self.sigma, atss_k_ratio=self.atss_k_ratio)
