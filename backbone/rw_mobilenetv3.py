import functools

import torch

import backbone.outlet.pytorch_image_models.timm as outlet


def feature_strides():
	return [8, 16, 32]


def MobileNetV3_Large(**kwargs):
	m = outlet.models.mobilenetv3_large_100(
		features_only=True,
		out_indices=(2, 3, 4),
		feature_location="bottleneck"
		**kwargs)

	m.feature_strides = feature_strides

	return m

def MobileNetV3_Small(**kwargs):
	m = outlet.models.mobilenetv3_small_100(
		features_only=True,
		out_indices=(1, 2, 3),
		feature_location="bottleneck",
		**kwargs)

	m.feature_strides = feature_strides

	return m
