import albumentations as A

from transform.to_tensor import ToTensor
from transform.convert_bbox_format import BboxFormatConvert

from processing.get_aug import GetAug


class Pipeline(GetAug):
	def __init__(self, size, mean, std, bbox_format='albumentations'):
		super(Pipeline, self).__init__([
			A.RandomContrast(),
			A.RandomGamma(),
			A.HueSaturationValue(sat_shift_limit=50, p=0.6),
			A.CLAHE(),
			A.HorizontalFlip(),
			A.Cutout(p=0.5),
			A.RandomSizedBBoxSafeCrop(size[1], size[0], p=0.8),
			A.Resize(size[1], size[0]),
			A.Normalize(),
			BboxFormatConvert(bbox_format, 'albumentations'),
			ToTensor()
		], bbox_format)


