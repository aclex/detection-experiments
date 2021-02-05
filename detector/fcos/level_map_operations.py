import torch


class LevelMapOperations:
	@staticmethod
	def _create_level_reg_maps(stride, image_size):
		r = torch.arange(0., 1., float(stride) / image_size)

		my, mx = torch.meshgrid(r, r)

		return mx.unsqueeze(-1), my.unsqueeze(-1)

	@staticmethod
	def split_joint_tensor(x, num_classes):
		return torch.split(x, [4, 1, num_classes], dim=-1)
