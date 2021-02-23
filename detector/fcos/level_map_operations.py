import torch


class LevelMapOperations:
	@staticmethod
	def _create_level_reg_maps(stride, image_size):
		r = torch.arange(0, image_size, stride)

		my, mx = torch.meshgrid(r, r)

		mx = mx.to(dtype=torch.float32) / image_size
		my = my.to(dtype=torch.float32) / image_size

		return mx.unsqueeze(-1), my.unsqueeze(-1)

	@staticmethod
	def split_joint_tensor(x, num_classes, with_centerness=True):
		if with_centerness:
			return torch.split(x, [4, 1, num_classes], dim=-1)
		else:
			return torch.split(x, [4, num_classes], dim=-1)
