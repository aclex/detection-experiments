import albumentations as A

from transform.to_tensor import ToTensor
from transform.convert_bbox_format import ConvertBboxFormat


class GetAug(A.Compose):
    def __init__(self, aug, bbox_format='albumentations',
                 min_area=0., min_visibility=0.):
	    super(GetAug, self).__init__(
            aug,
            A.BboxParams(format=bbox_format, min_area=min_area,
						 min_visibility=min_visibility,
                         label_fields=['category_id']))


class TrainAugmentation(GetAug):
    def __init__(self, size, mean, std, bbox_format='albumentations'):
        super(TrainAugmentation, self).__init__([
            A.RandomContrast(),
            A.RandomGamma(),
            A.HueSaturationValue(sat_shift_limit=50, p=0.6),
            A.CLAHE(),
            A.ShiftScaleRotate(rotate_limit=0),
            A.HorizontalFlip(),
            A.Cutout(p=0.5),
            A.RandomSizedBBoxSafeCrop(size[1], size[0], p=0.8),
            A.Resize(size[1], size[0]),
            A.Normalize(),
            ConvertBboxFormat(bbox_format, 'albumentations'),
            ToTensor()
        ], bbox_format)


class TestTransform(GetAug):
    def __init__(self, size, mean, std, bbox_format='albumentations'):
        super(TestTransform, self).__init__([
            A.Resize(size[1], size[0]),
            A.Normalize(),
            ConvertBboxFormat(bbox_format, 'albumentations'),
            ToTensor()
        ], bbox_format)


class PredictionTransform(GetAug):
    def __init__(self, size, mean, std, bbox_format='albumentations'):
        super(PredictionTransform, self).__init__([
            A.Resize(size[1], size[0]),
            A.Normalize(),
            ToTensor()
        ], bbox_format)
