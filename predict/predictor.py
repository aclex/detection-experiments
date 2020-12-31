import torch

from torchvision.ops.boxes import batched_nms

import processing.predict


class Predictor:
	def __init__(
			self, arch, net,
			nms_method=None, iou_threshold=0.45,
			filter_threshold=0.01, candidate_size=200, sigma=0.5,
			device="cpu"):
		self.net = net
		self.transform = processing.predict.Pipeline(
			[arch.image_size] * 2,
			arch.image_mean,
			arch.image_std)

		self.iou_threshold = iou_threshold
		self.filter_threshold = filter_threshold
		self.candidate_size = candidate_size
		self.nms_method = nms_method

		self.sigma = sigma

		self.device = device

		self.net.to(self.device)
		self.net.eval()

	def predict(self, image, top_k=-1, prob_threshold=None):
		cpu_device = torch.device("cpu")
		height, width, _ = image.shape
		image = self.transform(image=image, bboxes=[], category_id=[])["image"]
		images = image.unsqueeze(0)
		images = images.to(self.device)
		with torch.no_grad():
			scores, boxes = self.net.forward(images)
		boxes = boxes[0]
		scores = scores[0]
		if not prob_threshold:
			prob_threshold = self.filter_threshold
		# this version of nms is slower on GPU, so we move data to CPU.
		boxes = boxes.to(cpu_device)
		scores = scores.to(cpu_device)

		probs, labels = scores.max(dim=-1)

		# drop background detections
		non_bg = labels.nonzero(as_tuple=False).squeeze(dim=-1)

		boxes = boxes[non_bg]
		probs = probs[non_bg]
		labels = labels[non_bg]

		prefilter_mask = probs >= prob_threshold

		boxes = boxes[prefilter_mask]
		probs = probs[prefilter_mask]
		labels = labels[prefilter_mask]

		keep = batched_nms(boxes, probs, labels, self.iou_threshold)

		boxes = boxes[keep].detach()
		probs= probs[keep]
		labels= labels[keep]

		if len(boxes) == 0:
			return torch.tensor([]), torch.tensor([]), torch.tensor([])

		boxes[:, 0] *= width
		boxes[:, 1] *= height
		boxes[:, 2] *= width
		boxes[:, 3] *= height

		return (boxes.round().int(), labels, probs)
