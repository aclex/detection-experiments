import torch
import torch.nn.functional as F

from detector.ssd.utils import box_utils


class ToPredictions(torch.nn.Module):
	def __init__(self, priors, center_variance, size_variance):
		super(ToPredictions, self).__init__()

		self.priors = priors
		self.center_variance = center_variance
		self.size_variance = size_variance

	def forward(self, confidences, locations):
		confidences = F.softmax(confidences, dim=2)
		boxes = box_utils.convert_locations_to_boxes(
			locations,
			self.priors.to(device=locations.device),
			self.center_variance,
			self.size_variance)
		boxes = box_utils.center_form_to_corner_form(boxes)

		return confidences, boxes
