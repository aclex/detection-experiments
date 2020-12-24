import os
import json

import torch

from torch import nn

import arch.default_settings

import detector.ssd.config as config

from detector.ssd import ssd
from detector.ssd.multibox_loss import MultiboxLoss

from arch.core_settings import CoreSettings
from arch.compound_scaling import CompoundScaling

from arch.parse import (
	parse_backbone,
	parse_fpn_class,
	parse_conv,
	parse_norm,
	parse_act)


class SSD(CoreSettings):
	def __init__(self, config):
		super(SSD, self).__init__(config)

		self.backbone_class = parse_backbone(self.settings["backbone"])

	def build(
			self, num_classes, pretrained_backbone=False,
			batch_size=1, inference=False):
		backbone = self.backbone_class(pretrained=pretrained_backbone)

		ctor = ssd.SSDInference if inference else ssd.SSD

		return ctor(num_classes, backbone, self.name, batch_size, config)

	def loss(self, net, device=None):
		priors = config.priors.to(device=device, dtype=torch.float32)
		return MultiboxLoss(
			priors, iou_threshold=0.5, neg_pos_ratio=3,
			center_variance=0.1, size_variance=0.2)

	def mapper(self, net, device=None):
		return nn.Identity()
