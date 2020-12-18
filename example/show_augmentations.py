from matplotlib import pyplot as plt

from dataset.voc import VOCDetection

import albumentations as A

import cv2


BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(
		img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
	x_min, y_min, x_max, y_max = bbox
	x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
	cv2.rectangle(
		img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
	class_name = class_idx_to_name[class_id]
	((text_width, text_height), _) = cv2.getTextSize(
		class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
	cv2.rectangle(
		img,
		(x_min, y_min - int(1.3 * text_height)),
		(x_min + text_width, y_min), BOX_COLOR, -1)
	cv2.putText(
		img, class_name,
		(x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
		TEXT_COLOR, lineType=cv2.LINE_AA)

	return img


def visualize(annotations, category_id_to_name):
	img = annotations['image'].copy()
	for idx, bbox in enumerate(annotations['bboxes']):
		print("bbox:", bbox, "label:", annotations['category_id'][idx])
		img = visualize_bbox(
			img, bbox, annotations['category_id'][idx], category_id_to_name)
	plt.figure(figsize=(12, 12))
	plt.imshow(img)


def get_aug(aug, min_area=0., min_visibility=0.):
	return A.Compose(aug, A.BboxParams(
		format='pascal_voc', min_area=min_area,
		min_visibility=min_visibility, label_fields=['category_id']))


voc = VOCDetection(root="/mnt/dataset/vision/VOC/VOCdevkit", year="2012")
d = voc[3]

aug = get_aug([
	A.RandomContrast(),
	A.RandomGamma(),
	A.CLAHE(),
	A.Resize(300, 300)
])

augmented = aug(**d)

visualize(augmented, voc.class_names)

# plt.imshow(d["image"])

plt.show()
