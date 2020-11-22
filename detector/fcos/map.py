import torch

import math

from torch import nn

class Mapper(nn.Module):
	def __init__(self, strides, image_size):
		pass

	def forward(self, x):
		return None

	@classmethod
	def _calc_level_thresholds(cls, strides, image_size):
		return (
			cls._calc_level_thresholds_axis(strides, image_size[0]),
			cls._calc_level_thresholds_axis(strides, image_size[1]))

	@staticmethod
	def _calc_level_thresholds_axis(strides, axis_size):
		result = []

		last_size = axis_size
		for i in range(len(strides) - 1, -1, -1):
			s = strides[i]

			th_max = math.ceil(last_size / s)
			th_min = max(th_max - 4, 1)

			assert th_max > th_min

			last_size = th_min * s

			result.append((th_min, th_max))

		return tuple(result[::-1])
