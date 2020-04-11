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
                A.CLAHE(),
                A.HorizontalFlip(),
                A.Resize(*size),
                A.Normalize(),
                ConvertBboxFormat('pascal_voc', 'albumentations'),
                ToTensor()
            ], bbox_format)


class TestTransform(GetAug):
    def __init__(self, size, mean, std, bbox_format='albumentations'):
        super(TestTransform, self).__init__([
                A.Resize(*size),
                A.Normalize(),
                ConvertBboxFormat('pascal_voc', 'albumentations'),
                ToTensor()
            ], bbox_format)


class PredictionTransform(GetAug):
    def __init__(self, size, mean, std, bbox_format='albumentations'):
        super(PredictionTransform, self).__init__([
                A.Resize(*size),
                A.Normalize(),
                ConvertBboxFormat('pascal_voc', 'albumentations'),
                ToTensor()
            ], bbox_format)
