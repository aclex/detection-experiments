import math
import torch

from torchvision.ops.boxes import box_iou

def box_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
	"""
	Return complete intersection-over-union of boxes.

		As per "Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation" paper by Zheng et al.
	Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

	Arguments:
		boxes1 (Tensor[N, 4])
		boxes2 (Tensor[M, 4])

	Returns:
		ciou (Tensor[N, M]): the NxM matrix containing the pairwise CIoU values for every element in boxes1 and boxes2
	"""
	iou = box_iou(boxes1, boxes2)

	S = 1 - iou

	c_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
	c_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

	c_wh = (c_rb - c_lt).clamp(min=0)  # [N,M,2]

	p = (boxes1[:, :2] + boxes1[:, 2:]) / 2
	p_gt = (boxes2[:, :2] + boxes2[:, 2:]) / 2

	ro = (p[:, 0] - p_gt[:, 0]) ** 2 + (p[:, 1] - p_gt[:, 1]) ** 2
	c2 = c_wh[..., 0] ** 2 + c_wh[..., 1] ** 2

	D = ro / c2

	wh = boxes1[:, 2:] - boxes1[:, :2]
	wh_gt = boxes2[:, 2:] - boxes2[:, :2]

	V = (2 * (
			torch.arctan(wh_gt[:, None, 0] / wh_gt[:, None, 1]) -
			torch.arctan(wh[:, 0] / wh[:, 1])) / math.pi) ** 2

	alpha = torch.where(iou >= 0.5, V / (S + V), torch.tensor([0.]))

	return S + D + alpha * V
