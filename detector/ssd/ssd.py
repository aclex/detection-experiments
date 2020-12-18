import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from detector.ssd.utils import box_utils

from nn.separable_conv_2d import SeparableConv2d

from fpn.extension import Extension
from detector.ssd.to_predictions import ToPredictions


class SSD(nn.Module):
	def __init__(self, num_classes, backbone, arch_name,
	             batch_size=None, config=None):
		"""Compose a SSD model using the given components.
		"""
		super(SSD, self).__init__()

		self.num_classes = num_classes
		self.backbone = backbone
		self.arch_name = arch_name
		self.batch_size = batch_size # to ease the inference model

		feature_channels = self.backbone.feature_channels()

		self.extras = Extension(
			bootstrap_channels=feature_channels[-1],
			out_channels=[512, 256, 256, 64],
			conv=SeparableConv2d)

		self.classification_headers = nn.ModuleList([
			SeparableConv2d(in_channels=feature_channels[-2],
			                out_channels=6 * num_classes,
			                kernel_size=3, padding=1),
			SeparableConv2d(in_channels=feature_channels[-1],
			                out_channels=6 * num_classes,
			                kernel_size=3, padding=1),
			SeparableConv2d(in_channels=512, out_channels=6 * num_classes,
			                kernel_size=3, padding=1),
			SeparableConv2d(in_channels=256, out_channels=6 * num_classes,
			                kernel_size=3, padding=1),
			SeparableConv2d(in_channels=256, out_channels=6 * num_classes,
			                kernel_size=3, padding=1),
			nn.Conv2d(in_channels=64, out_channels=6 * num_classes,
			          kernel_size=1),
		])

		self.regression_headers = nn.ModuleList([
			SeparableConv2d(in_channels=feature_channels[-2],
			                out_channels=6 * 4,
			                kernel_size=3, padding=1, onnx_compatible=False),
			SeparableConv2d(in_channels=feature_channels[-1],
			                out_channels=6 * 4, kernel_size=3,
			                padding=1, onnx_compatible=False),
			SeparableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3,
			                padding=1, onnx_compatible=False),
			SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3,
			                padding=1, onnx_compatible=False),
			SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3,
			                padding=1, onnx_compatible=False),
			nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
		])

		self.config = config

	def forward(self, x):
		confidences = []
		locations = []

		cs = self.backbone.forward(x)
		cs = cs[-2:]

		for i, c in enumerate(cs):
			confidence, location = self.compute_header(i, c)
			x = c
			confidences.append(confidence)
			locations.append(location)

		extra_x = self.extras.forward(x)

		header_index = i + 1

		for ex in extra_x:
			confidence, location = self.compute_header(header_index, ex)
			header_index += 1
			confidences.append(confidence)
			locations.append(location)

		confidences = torch.cat(confidences, 1)
		locations = torch.cat(locations, 1)

		return confidences, locations

	def compute_header(self, i, x):
		batch_size = self.batch_size or x.size(0)

		confidence = self.classification_headers[i](x)
		confidence = confidence.permute(0, 2, 3, 1).contiguous()
		confidence = confidence.reshape(batch_size, -1, self.num_classes)

		location = self.regression_headers[i](x)
		location = location.permute(0, 2, 3, 1).contiguous()
		location = location.reshape(batch_size, -1, 4)

		return confidence, location

	def load_backbone_weights(self, path):
		self.backbone.load_state_dict(
			torch.load(path, map_location=lambda storage, loc: storage),
			strict=True)

	def freeze_backbone(self):
		for p in self.backbone.parameters():
			p.requires_grad = False


class SSDInference(SSD):
	def __init__(self, num_classes, backbone, arch_name,
	             batch_size=None, config=None):
		super(SSDInference, self).__init__(num_classes, backbone, arch_name,
		                                   batch_size, config)

		self.to_predictions = ToPredictions(self.config.priors,
		                                    self.config.center_variance,
		                                    self.config.size_variance)

	def forward(self, x):
		confidences, locations = super(SSDInference, self).forward(x)
		confidences, boxes = self.to_predictions.forward(confidences, locations)

		return confidences, boxes
