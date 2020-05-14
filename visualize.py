import sys
import argparse

from tqdm import tqdm

import cv2

from transform.convert_bbox_format import BboxFormatConvert


from dataset.voc import VOCDetection
from dataset.coco import CocoDetection


def draw_box(frame, box, label, class_names):
	box = box[:]
	cv2.rectangle(frame,
	              (int(box[0]), int(box[1])),
	              (int(box[2]), int(box[3])),
	              (255, 255, 0), 4)

	label = f"{class_names[label]}"
	cv2.putText(frame, label,
	            (int(box[0]) + 20, int(box[1]) + 40),
	            cv2.FONT_HERSHEY_SIMPLEX,
	            1,  # font scale
	            (255, 0, 255),
	            2)  # line type


def main():
	parser = argparse.ArgumentParser(
		description="Dataset visualization utility")

	parser.add_argument('--dataset-style', type=str, required=True,
	                    help="style of dataset "
	                    "(supported are 'pascal-voc' and 'coco')")

	parser.add_argument('--image-set', type=str, default="test",
	                    help='image set (annotation file basename for COCO) '
	                    'to use for evaluation')

	parser.add_argument("--dataset", type=str,
	                    help="dataset directory path")

	args = parser.parse_args()

	if args.dataset_style == 'pascal-voc':
		dataset = VOCDetection(root=args.dataset,
		                       image_set=args.image_set)
		transform = None

	elif args.dataset_style == 'coco':
		dataset = CocoDetection(root=args.dataset,
		                        ann_file="%s.json" % args.image_set)
		transform = BboxFormatConvert(source_format='coco',
		                              target_format='pascal_voc')


	for sample in tqdm(dataset):
		if transform:
			sample = transform(**sample)

		image = sample["image"]

		for bbox, label in zip(sample["bboxes"], sample["category_id"]):
			draw_box(image, bbox, label, dataset.class_names)

		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		cv2.imshow("visualize", image)

		if cv2.waitKey(0) == 0x1b:
			break


if __name__ == "__main__":
	try:
		main()

	except KeyboardInterrupt:
		sys.exit(0)
