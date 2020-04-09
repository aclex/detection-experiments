import torch

from albumentations.pytorch.transforms import ToTensorV2


class ToTensor(ToTensorV2):
	def __init__(self, always_apply=True, p=1.0):
		super(ToTensor, self).__init__(always_apply, p)

	@property
	def targets(self):
		super_targets = super(ToTensor, self).targets

		super_targets.update({
			"bboxes": self.apply_to_bboxes,
			"category_id": self.apply_to_category_id
		})

		return super_targets

	def apply_to_bboxes(self, bboxes, **params):
		return torch.tensor(bboxes)

	def apply_to_category_id(self, category_id, **params):
		return torch.tensor(category_id)
