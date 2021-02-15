from timm.models.efficientnet_builder import *
from timm.models.efficientnet import EfficientNetFeatures, _cfg


EDITIONS = {
	'a' : (0.86, 1.0, 1.2),
	'b' : (0.84, 0.75, 1.1),
	'c' : (0.825, 0.54, 0.85),
	'd' : (0.68, 0.54, 0.695),
	'e' : (0.475, 0.51, 0.6)
}


def feature_strides():
	return [8, 16, 32]


def TinyNet(edition, **kwargs):
	kwargs.pop('pretrained')
	r, w, d = EDITIONS.get(edition.casefold(), (1., 1., 1.))

	"""Creates a TinyNet model.
	"""
	arch_def = [
		['ds_r1_k3_s1_e1_c16_se0.25'], ['ir_r2_k3_s2_e6_c24_se0.25'],
		['ir_r2_k5_s2_e6_c40_se0.25'], ['ir_r3_k3_s2_e6_c80_se0.25'],
		['ir_r3_k5_s1_e6_c112_se0.25'], ['ir_r4_k5_s2_e6_c192_se0.25'],
		['ir_r1_k3_s1_e6_c320_se0.25'],
	]
	model_kwargs = dict(
		block_args=decode_arch_def(arch_def, d, depth_trunc="round"),
		stem_size=32,
		fix_stem=True,
		channel_multiplier=w,
		act_layer=Swish,
		norm_kwargs=resolve_bn_args(kwargs),
		out_indices=(2, 3, 4),
		feature_location="bottleneck",
		**kwargs
	)

	m = EfficientNetFeatures(**model_kwargs)

	hw = int(224 * r)
	m.default_cfg =_cfg(input_size=(3, hw, hw))

	m.feature_strides = feature_strides

	return m
