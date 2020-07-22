import backbone.outlet.ghostnet.pytorch.ghostnet as outlet

#  from backbone.outlet.mobilenetv3.mobilenetv3 import Block, hswish

from backbone.feature_hook import FeatureHook


class GhostNet(outlet.GhostNet):
	BOTTLENECK_NO = 6
	def __init__(self, **kwargs):
		kwargs.pop('pretrained')

		p = outlet.ghostnet()
		super(GhostNet, self).__init__(p.cfgs, **kwargs)

		self.c4 = FeatureHook()

		self.blocks[self.BOTTLENECK_NO].register_forward_hook(self.c4)

	def forward(self, x):
		
		out = self.act1(self.bn1(self.conv_stem(x)))
		out = self.blocks(out)

		return self.c4.output, out # C4 and C5

	def feature_channels(self, idx=None):
		c4_block = self.blocks[self.BOTTLENECK_NO][-1].ghost2
		result = [
			c4_block.primary_conv[1].num_features +
				c4_block.cheap_operation[1].num_features,
			self.blocks[-1][0].bn1.num_features
		]

		if isinstance(idx, int):
			return result[idx]

		return result
