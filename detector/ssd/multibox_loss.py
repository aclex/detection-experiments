import torch
import torch.nn as nn
import torch.nn.functional as F

import detector.ssd.config as config

from .match_prior import MatchPrior
from .utils import box_utils


class MultiboxLoss(nn.Module):
	def __init__(self, priors, iou_threshold, neg_pos_ratio,
	             center_variance, size_variance):
		"""Implement SSD Multibox Loss.

		Basically, Multibox loss combines classification loss
		 and Smooth L1 regression loss.
		"""
		super(MultiboxLoss, self).__init__()

		self.iou_threshold = iou_threshold
		self.neg_pos_ratio = neg_pos_ratio
		self.center_variance = center_variance
		self.size_variance = size_variance

		self.match_prior = MatchPrior(priors,
		                              config.center_variance,
		                              config.size_variance,
		                              iou_threshold=0.5)

		self.priors = priors

	def forward(self, x, y):
		confidence, predicted_locations = x
		gt_boxes, labels = y
		locations, labels = self.match_prior(gt_boxes, labels)

		return self._apply(confidence, predicted_locations, labels, locations)

	def _apply(self, confidence, predicted_locations, labels, gt_locations):
		"""Compute classification loss and smooth l1 loss.

		Args:
			confidence (batch_size, num_priors, num_classes): class predictions.
			locations (batch_size, num_priors, 4): predicted locations.
			labels (batch_size, num_priors): real labels of all the priors.
			boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
		"""
		num_classes = confidence.size(2)
		with torch.no_grad():
			# derived from cross_entropy=sum(log(p))
			loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
			mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

		confidence = confidence[mask, :]
		classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], reduction='sum')

		pos_mask = labels > 0
		predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
		gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
		smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')

		result = {
			"reg": smooth_l1_loss,
			"cls": classification_loss
		}

		num_pos = gt_locations.size(0)
		if num_pos > 0:
			result["reg"] /= num_pos
			result["cls"] /= num_pos

		result.update({"total": (result["reg"] + result["cls"])})

		return result

	@staticmethod
	def _combine(reg_loss, cls_loss):
		return reg_loss + cls_loss

