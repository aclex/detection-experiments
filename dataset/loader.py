from dataset.voc import VOCDetection
from dataset.coco import CocoDetection
from dataset.widerface import WIDERFace

def load(style, path, image_set, transform=None):
	if style == 'pascal-voc':
		return VOCDetection(
			root=path,
			image_set=image_set,
			transform=transform)
	elif style == 'coco':
		return CocoDetection(
			root=path,
			ann_file="%s.json" % image_set,
			transform=transform)
	elif style == 'widerface':
		return WIDERFace(
			root=path,
			split=image_set,
			transform=transform)
	else:
		raise RuntimeError("Dataset style %s is not supported" % style)

def bbox_format(style):
	if style == 'pascal-voc':
		return VOCDetection.bbox_format
	elif style == 'coco':
		return CocoDetection.bbox_format
	elif style == 'widerface':
		return WIDERFace.bbox_format
	else:
		raise RuntimeError("Dataset style %s is not supported" % style)

