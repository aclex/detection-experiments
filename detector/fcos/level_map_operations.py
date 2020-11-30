import torch


class LevelMapOperations:
	@staticmethod
	def _create_level_reg_maps(stride, image_size):
		r = torch.arange(0, image_size, stride)

		my, mx = torch.meshgrid(r, r)

		return mx.unsqueeze(-1), my.unsqueeze(-1)
