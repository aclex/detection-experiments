import torch

from collections import OrderedDict

from torch import nn


class SoftMaxWeightedSum(nn.Module):
	def __init__(self, op_number=2):
		super(SoftMaxWeightedSum, self).__init__()
		shape = (op_number, 1, 1, 1, 1)
		self.weights = nn.Parameter(torch.ones(shape), requires_grad=True)

	def forward(self, x):
		return torch.sum(torch.softmax(self.weights, dim=0) * x, dim=0)


class HardWeightedSum(nn.Module):
	def __init__(self, op_number=2, act=nn.ReLU, eps=1e-4):
		super(HardWeightedSum, self).__init__()
		shape = (op_number, 1, 1, 1, 1)
		self.weights = nn.Parameter(torch.ones(shape), requires_grad=True)

		self.act = act()
		self.eps = eps

	def forward(self, x):
		weights_num = self.act(self.weights)
		weights_denom = torch.sum(weights_num) + self.eps

		return torch.sum(weights_num * x / weights_denom, dim=0)


class BiFPNLayer(nn.Module):
	def __init__(
			self, num_channels, scales,
			upsampling_mode='nearest',
			conv=nn.Conv2d, norm=nn.BatchNorm2d,
			act=nn.ReLU, pooling=nn.MaxPool2d,
			weighted_sum=HardWeightedSum):
		super(BiFPNLayer, self).__init__()

		self.num_levels = len(scales) + 1

		self.upsamplers = nn.ModuleList()

		for i in range(self.num_levels - 1):
			self.upsamplers.append(nn.Upsample(
				scale_factor=scales[i], mode=upsampling_mode))

		self.inner = nn.ModuleList()

		for i in range(self.num_levels - 1):
			node = nn.Sequential(
				weighted_sum(2),
				conv(
					num_channels, num_channels, kernel_size=3,
					padding=1, bias=(norm is None)),
				norm(num_features=num_channels) if norm else nn.Identity(),
				act())

			self.inner.append(node)

		self.downsamplers = nn.ModuleList()

		for i in range(self.num_levels - 1):
			self.downsamplers.append(pooling(kernel_size=scales[i]))

		self.outer = nn.ModuleList()

		for i in range(self.num_levels - 1):
			node = nn.Sequential(
				weighted_sum(3 if i != self.num_levels - 2 else 2),
				conv(
					num_channels, num_channels, 3,
					padding=1, bias=(norm is None)),
				norm(num_features=num_channels) if norm else nn.Identity(),
				act())

			self.outer.append(node)

	def forward(self, x):
		ptd = []
		for i in range(self.num_levels - 1, -1, -1):
			if i == self.num_levels - 1:
				p = x[i]
			else:
				f = torch.stack([x[i], self.upsamplers[i].forward(ptd[0])])
				p = self.inner[i].forward(f)

			ptd.insert(0, p)

		out = []
		for i in range(self.num_levels):
			if i == 0:
				p = ptd[i]
			else:
				ra = self.downsamplers[i - 1].forward(out[-1])

				if i < self.num_levels - 1:
					f = torch.stack([x[i], ptd[i], ra])
				else:
					f= torch.stack([x[i], ra])

				p = self.outer[i - 1].forward(f)

			out.append(p)

		return out


class BiFPN(nn.Module):
	def __init__(
			self, feature_channels, feature_strides, out_channels, num_layers=1,
			upsampling_mode='nearest',
			conv=nn.Conv2d, norm=nn.BatchNorm2d,
			act=nn.ReLU, pooling=nn.MaxPool2d, weighted_sum=HardWeightedSum):
		super(BiFPN, self).__init__()

		assert len(feature_channels) == len(feature_strides)

		self.out_channels = out_channels
		self.num_levels = len(feature_channels)

		self.feature_channels = feature_channels
		self.feature_strides = feature_strides

		self.scales = self.calc_scales(self.feature_strides)

		self.input_transforms = nn.ModuleList()

		for i in range(self.num_levels):
			node = nn.Sequential(
				conv(
					self.feature_channels[i], out_channels, kernel_size=1,
					bias=(norm is None)),
				norm(num_features=out_channels) if norm else nn.Identity(),
				act())

			self.input_transforms.append(node)

		self.layers = nn.Sequential(OrderedDict([
			("bifpn_layer%d" % i, BiFPNLayer(
				self.out_channels, self.scales, upsampling_mode,
				conv=conv, norm=norm, act=act, pooling=pooling,
				weighted_sum=weighted_sum))
			for i in range(num_layers)
		]))

	def calc_scales(self, feature_strides):
		scales = []

		for i in range(1, len(feature_strides)):
			scales.append(feature_strides[i] // feature_strides[i - 1])

		assert len(scales) == len(feature_strides) - 1

		return scales

	def forward(self, x):
		assert len(x) == self.num_levels

		tx = []
		for i in range(self.num_levels):
			tx.append(self.input_transforms[i](x[i]))

		return self.layers.forward(tx)
