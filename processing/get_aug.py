import albumentations as A

from transform.to_tensor import ToTensor
from transform.convert_bbox_format import BboxFormatConvert


class GetAug(A.Compose):
	def __init__(
			self, aug, bbox_format='albumentations',
			min_area=0., min_visibility=0.):
		super(GetAug, self).__init__(
			aug,
			A.BboxParams(
				format=bbox_format, min_area=min_area,
				min_visibility=min_visibility,
				label_fields=['category_id']))
