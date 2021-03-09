import types

import torch

import timm as outlet


def feature_strides():
	return [8, 16, 32]

def feature_channels(self):
	return self.feature_info.channels()


def Timm(**kwargs):
	model_name = kwargs.pop('model')
	pretrained = kwargs.pop('pretrained')
	print(
		"Loading \"%s\" model from `pytorch-image-models` collection..." %
		model_name)
	m = outlet.create_model(
		model_name,
		pretrained=pretrained,
		features_only=True,
		out_indices=(2, 3, 4),
		#  feature_location="bottleneck",
		**kwargs)

	m.feature_strides = feature_strides
	m.feature_channels = types.MethodType(feature_channels, m)

	return m
