from torch import nn


class SoftMaxWeightedSum(nn.Module):
	def __init__(self, op_number=2):
		super(SoftMaxWeightedSum, self).__init__()
		self.weights = nn.Parameter(torch.ones(op_number), requires_grad=True)

	def forward(self, x):
		return torch.softmax(self.weights, dim=0) * x


class HardWeightedSum(nn.Module):
	def __init__(self, op_number=2, act=nn.Relu, eps=1e-4):
		super(HardWeightedSum, self).__init__()
		self.weights = nn.Parameter(torch.ones(op_number), requires_grad=True)

		self.act = act
		self.eps = eps

	def forward(self, x):
		weights_num = self.act(self.weights)
		weights_denom = torch.sum(weights_num) + self.eps

		return weights_num * x / weights_denom


class BiFPNLayer(nn.Module):
	def __init__(
			self, num_channels, scales,
			upsampling_mode='nearest',
			conv=nn.Conv2d, norm=nn.BatchNorm2d,
			act=nn.ReLU, pooling=nn.MaxPool2d):
		super(BiFPNLayer, self).__init__()

		self.num_levels = len(scales) + 1

		self.upsamplers = nn.ModuleList()

		for i in range(self.num_levels - 1):
			self.upsamplers.append(nn.Upsample(
				scale_factor=scales[i], mode=upsampling_mode))

		self.inner = nn.ModuleList()

		for i in range(self.num_levels - 1):
			node = nn.Sequential([
				weighted_sum(2),
				conv(
					num_channels, num_channels, 3,
					padding=1, bias=(norm is None)),
				norm_layer() if norm else nn.Identity(),
				act()
			]
			self.inner.append(node)

		self.downsamplers = nn.ModuleList()

		for i in range(self.num_levels - 1):
			self.downsamplers.append(pooling(kernel_size=scales[i]))

		self.outer = nn.ModuleList()

		for i in range(self.num_levels - 2):
			node = nn.Sequence([
				weighted_sum(3 if i != self.num_levels - 2 else 2),
				conv(
					num_channels, num_channels, 3,
					padding=1, bias=(norm_layer is None)),
				norm_layer() if norm_layer else nn.Identity(),
				act()
			])
			self.outer.append(node)

	def forward(self, x):
		ptd = []
		for i in range(self.num_levels - 1, 0, -1):
			if i == self.num_levels - 1:
				p = x[i]
			else:
				p = self.inner[i - 1](x[i], self.upsamplers[i - 1](ptd[0])

			ptd.insert(0, p)

		out = []
		for i in range(self.num_levels):
			if i == 0:
				p = ptd[i]
			else i < len(x) - 1:
				p = self.outer[i](
					x[i], ptd[i], self.downsamples[i - 1](out[-1]))
			else:
				p = self.outer[i](x[i], self.downsamples(out[-1]))

			out.append(p)

		return out


class BiFPN(nn.Module):
	def __init__(
			self, feature_channels, feature_strides, out_channels, num_layers=1,
			upsampling_mode='nearest',
			conv=nn.Conv2d, norm=nn.BatchNorm2d,
			act=nn.ReLU, pooling=nn.MaxPool2d):
		super(BiFPN, self).__init__()

		assert len(feature_channels) == len(feature_strides)

		self.num_levels = len(feature_channels)

		self.feature_channels = feature_channels
		self.feature_strides = feature_strides

		self.scales = self.calc_scales(self.feature_strides)

		self.input_transforms = nn.ModuleList()

		for i in range(self.num_levels):
			node = nn.Sequential([
				conv(
					self.feature_channels[i], out_channels, 1,
					bias=(norm is None)),
				norm() if norm else nn.Identity(),
				act()
			])

			self.input_transforms.append(node)

		self.layers = nn.Sequential([
			BiFPNLayer(
				self.out_channels, scales, upsampling_mode,
				conv=conv, norm=norm, act=act, pooling=pooling)
			for _ in range(num_layers)
		])

	def calc_scales(feature_strides):
		scales = []

		for i in range(1, len(feature_strides)):
			scales.append(feature_strides[i] / feature_strides[i - 1])

		assert len(scales) == len(feature_strides) - 1

		return scales

	def forward(self, x):
		assert len(x) == self.num_levels

		tx = []
		for i in range(self.num_levels):
			tx.append(self.input_transforms(x[i]))

		return self.layers(tx)
