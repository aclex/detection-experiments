import numpy as np

import torch

from .utils import box_utils


class MatchPrior(object):
	def __init__(self, center_form_priors,
				 center_variance, size_variance, iou_threshold):
		self.center_form_priors = center_form_priors
		self.corner_form_priors = box_utils.center_form_to_corner_form(
			center_form_priors)
		self.center_variance = center_variance
		self.size_variance = size_variance
		self.iou_threshold = iou_threshold

	def __call__(self, gt_boxes, gt_labels):
		locations = []
		labels = []
		for b, l in zip(gt_boxes, gt_labels):
			loc, lb = self._apply_on_sample(b, l)

			locations.append(loc)
			labels.append(lb)

		locations = torch.stack(locations)
		labels = torch.stack(labels)

		return locations, labels

	def _apply_on_sample(self, gt_boxes, gt_labels):
		boxes, labels = box_utils.assign_priors(
			gt_boxes, gt_labels, self.corner_form_priors, self.iou_threshold)

		boxes = box_utils.corner_form_to_center_form(boxes)
		locations = box_utils.convert_boxes_to_locations(
			boxes, self.center_form_priors,
			self.center_variance, self.size_variance)

		return locations, labels
