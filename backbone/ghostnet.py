import backbone.outlet.ghostnet.ghostnet_pytorch.ghostnet as outlet

from backbone.feature_hook import FeatureHook


class GhostNet(outlet.GhostNet):
	C3_BOTTLENECK_NO = 4
	C4_BOTTLENECK_NO = 6
	def __init__(self, **kwargs):
		kwargs.pop('pretrained')

		p = outlet.ghostnet()
		super(GhostNet, self).__init__(p.cfgs, **kwargs)

		self.c3 = FeatureHook()
		self.c4 = FeatureHook()

		self.blocks[self.C3_BOTTLENECK_NO].register_forward_hook(self.c3)
		self.blocks[self.C4_BOTTLENECK_NO].register_forward_hook(self.c4)

	def forward(self, x):

		out = self.act1(self.bn1(self.conv_stem(x)))
		out = self.blocks(out)

		return self.c3.output, self.c4.output, out # C3, C4 and C5

	def feature_channels(self, idx=None):
		c3_block = self.blocks[self.C3_BOTTLENECK_NO][-1].ghost2
		c4_block = self.blocks[self.C4_BOTTLENECK_NO][-1].ghost2
		result = [
			c3_block.primary_conv[1].num_features +
				c3_block.cheap_operation[1].num_features,
			c4_block.primary_conv[1].num_features +
				c4_block.cheap_operation[1].num_features,
			self.blocks[-1][0].bn1.num_features
		]

		if isinstance(idx, int):
			return result[idx]

		return result

	def feature_strides(self):
		return [8, 16, 32]


class GhostNet075(GhostNet):
	def __init__(self, **kwargs):
		super().__init__(width=0.75, **kwargs)


class GhostNet050(GhostNet):
	def __init__(self, **kwargs):
		super().__init__(width=0.50, **kwargs)


class GhostNet025(GhostNet):
	def __init__(self, **kwargs):
		super().__init__(width=0.25, **kwargs)
