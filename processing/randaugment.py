import random

import albumentations as A

from transform.to_tensor import ToTensor
from transform.convert_bbox_format import BboxFormatConvert

from processing.get_aug import GetAug


class RandAugment:
	def __init__(self, n, m):
		self.n = n
		self.m = m

		m_ratio = self.m / 30.0
		self.augment_list = (
			A.CLAHE(always_apply=True),
			A.Equalize(always_apply=True),
			A.InvertImg(always_apply=True),
			A.Rotate(limit=30 * m_ratio, always_apply=True),
			A.Posterize(num_bits=int(4 * m_ratio), always_apply=True),
			A.Solarize(threshold=m_ratio, always_apply=True),
			A.RGBShift(
				r_shift_limit=110 * m_ratio,
				g_shift_limit=110 * m_ratio,
				b_shift_limit=110 * m_ratio,
				always_apply=True),
			A.HueSaturationValue(
				hue_shift_limit=20 * m_ratio,
				sat_shift_limit=30 * m_ratio,
				val_shift_limit=20 * m_ratio,
				always_apply=True),
			A.RandomContrast(limit=m_ratio, always_apply=True),
			A.RandomBrightness(limit=m_ratio, always_apply=True),
			#  A.Sharpen(always_apply=True), 0.1, 1.9),
			A.ShiftScaleRotate(
				shift_limit=0.3 * m_ratio,
				shift_limit_y=0,
				rotate_limit=0,
				always_apply=True),
			A.ShiftScaleRotate(
				shift_limit=0.3 * m_ratio,
				shift_limit_x=0,
				rotate_limit=0,
				always_apply=True),
			A.Cutout(
				num_holes=int(8 * m_ratio),
				always_apply=True),
			A.IAAAffine(shear=0.3 * m_ratio, always_apply=True))

		assert self.n <= len(self.augment_list)

	def __call__(self, *args, force_apply=False, **data):
		ops = random.choices(self.augment_list, k=self.n)

		for op in ops:
			data = op(force_apply=force_apply, **data)

		return data


class Pipeline(GetAug):
	def __init__(self, m, n, size, mean, std, bbox_format='albumentations'):
		super(Pipeline, self).__init__([
			A.Resize(size[1], size[0]),
			BboxFormatConvert(bbox_format, 'albumentations'),
			RandAugment(n, m),
			A.Normalize(mean=mean, std=std),
			ToTensor(),
		], bbox_format)
