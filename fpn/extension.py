from torch import nn


class Extension(nn.Module):
	def __init__(
			self, num_levels, in_channels, out_channels,
			kernel_sizes=None, paddings=None, strides=None,
			conv=nn.Conv2d):
		super(Extension, self).__init__()

		kernel_sizes = kernel_sizes or [3] * num_levels
		paddings = paddings or [1] * num_levels
		strides = strides or [2] * num_levels

		self.num_levels = num_levels

		self.levels = nn.ModuleList([
			conv(
				in_channels[i], out_channels[i], kernel_size=kernel_sizes[i],
				padding=paddings[i], stride=strides[i])
			for i in range(num_levels)
		])

	def forward(self, x):
		result = []

		out = x

		for i in range(self.num_levels):
			out = self.levels[i](out)
			result.append(out)

		return result
