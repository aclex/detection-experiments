import torch

from torch import nn

from fpn.extension import Extension


class Model(nn.Module):
	def __init__(
			self,
			backbone, fpn_builder, head_builder,
			num_levels, fpn_channels,
			conv=nn.Conv2d, norm=nn.BatchNorm2d, act=nn.ReLU):
		super(Model, self).__init__()

		self.backbone = backbone

		feature_channels = self.backbone.feature_channels()
		feature_strides = self.backbone.feature_strides()

		extra_levels = num_levels - len(feature_strides)
		extra_channels = [fpn_channels] * extra_levels

		if extra_levels > 0:
			self.extension = Extension(
				bootstrap_channels=feature_channels[-1],
				out_channels=extra_channels,
				conv=conv)

		extra_strides = []
		for i in range(extra_levels):
			extra_strides.append(
				(extra_strides[-1] if i > 0 else feature_strides[-1]) * 2)

		fpn_input_channels = feature_channels + extra_channels
		self.strides = feature_strides + extra_strides
		self.fpn = fpn_builder(
			feature_channels=fpn_input_channels,
			feature_strides=self.strides,
			out_channels=fpn_channels,
			conv=conv, norm=norm, act=act)

		self.head = head_builder(
			strides=self.strides)

	def forward(self, x):
		features = self.backbone.forward(x)

		if hasattr(self, "extension"):
			extras = self.extension.forward(features[-1])

			extra_features = features + extras
		else:
			extra_features = features

		fpn_out = self.fpn.forward(extra_features)
		out = self.head.forward(fpn_out)

		return out
