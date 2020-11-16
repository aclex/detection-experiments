from torch import nn


class Extension(nn.Module):
	def __init__(
			self, bootstrap_channels, out_channels,
			kernel_sizes=None, paddings=None, strides=None,
			conv=nn.Conv2d):
		super(Extension, self).__init__()

		self.num_levels = len(out_channels)

		kernel_sizes = kernel_sizes or [3] * self.num_levels
		paddings = paddings or [1] * self.num_levels
		strides = strides or [2] * self.num_levels

		self.levels = nn.ModuleList([
			conv(
				out_channels[i] if i > 0 else bootstrap_channels,
				out_channels[i], kernel_size=kernel_sizes[i],
				padding=paddings[i], stride=strides[i])
			for i in range(self.num_levels)
		])

	def forward(self, x):
		result = []

		out = x

		for i in range(self.num_levels):
			out = self.levels[i](out)
			result.append(out)

		return result
