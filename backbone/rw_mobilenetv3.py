import functools
import types

import torch

import timm as outlet


def feature_strides():
	return [8, 16, 32]

def feature_channels(self):
	return self.feature_info.channels()


def MobileNetV3_Large(**kwargs):
	m = outlet.models.mobilenetv3_large_100(
		features_only=True,
		out_indices=(2, 3, 4),
		feature_location="bottleneck",
		**kwargs)

	m.feature_strides = feature_strides
	#  m.feature_channels = types.MethodType(feature_channels, m)

	return m

def MobileNetV3_Small(**kwargs):
	m = outlet.models.mobilenetv3_small_100(
		features_only=True,
		out_indices=(1, 2, 3),
		feature_location="bottleneck",
		**kwargs)

	m.feature_strides = feature_strides
	#  m.feature_channels = types.MethodType(feature_channels, m)

	return m
