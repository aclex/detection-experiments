import math
import torch

from torchvision.ops.boxes import box_iou


def _calc_area(boxes: torch.Tensor) -> torch.Tensor:
	return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def _num_stab(t, eps=1e-9):
	return torch.where(t > 0., t, t.new_tensor([eps]))


def elementwise_iou(
		boxes1: torch.Tensor, boxes2: torch.Tensor, eps=1e-9) -> torch.Tensor:
	area1 = _calc_area(boxes1)
	area2 = _calc_area(boxes2)

	lti = torch.max(boxes1[..., :2], boxes2[..., :2])
	rbi = torch.min(boxes1[..., 2:], boxes2[..., 2:])

	whi = (rbi - lti).clamp(min=0)

	intersection = whi[..., 0] * whi[..., 1]
	union = area1 + area2 - intersection

	return intersection / _num_stab(union, eps)


def elementwise_ciou(
		boxes1: torch.Tensor, boxes2: torch.Tensor, eps=1e-9) -> torch.Tensor:
	iou = elementwise_iou(boxes1, boxes2, eps)

	S = 1 - iou

	c_lt = torch.min(boxes1[..., :2], boxes2[..., :2])  # [N, 2]
	c_rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])  # [N, 2]

	c_wh = (c_rb - c_lt).clamp(min=0)  # [N, 2]

	p = (boxes1[..., :2] + boxes1[..., 2:]) / 2
	p_gt = (boxes2[..., :2] + boxes2[..., 2:]) / 2

	ro = (p[..., 0] - p_gt[..., 0]) ** 2 + (p[..., 1] - p_gt[..., 1]) ** 2
	c2 = c_wh[..., 0] ** 2 + c_wh[..., 1] ** 2

	D = ro / c2

	wh = boxes1[..., 2:] - boxes1[..., :2]
	wh_gt = boxes2[..., 2:] - boxes2[..., :2]

	V = (
		2 * (
			torch.arctan(wh_gt[..., 0] / _num_stab(wh_gt[..., 1], eps)) -
			torch.arctan(wh[..., 0] / _num_stab(wh[..., 1], eps)))
		/ math.pi) ** 2

	S_V = S + V
	alpha = torch.where(
		iou >= 0.5, V / _num_stab(S + V, eps), torch.tensor([0.]).to(S.device))

	return S + D + alpha * V


def permutated_ciou(
		boxes1: torch.Tensor, boxes2: torch.Tensor, eps=1e-5) -> torch.Tensor:
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
			torch.arctan(
				wh_gt[:, None, 0] / _num_stab(wh_gt[:, None, 1], eps)) -
			torch.arctan(wh[:, 0] / _num_stab(wh[:, 1], eps))) / math.pi) ** 2

	alpha = torch.where(
		iou >= 0.5, V / _num_stab(S + V, eps), torch.tensor([0.]))

	return S + D + alpha * V


def box_ciou(
		boxes1: torch.Tensor, boxes2: torch.Tensor,
		mode=None, eps=1e-5) -> torch.Tensor:
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

	if mode is None:
		if boxes1.size(-2) == boxes2.size(-2):
			mode = "elementwise"
		else:
			mode = "permutated"

	if mode == "elementwise":
		return elementwise_ciou(boxes1, boxes2, eps)

	else: # "permutated"
		return permutated_ciou(boxes1, boxes2, eps)
