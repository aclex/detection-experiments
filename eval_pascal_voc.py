import sys
import argparse
import pathlib
import logging

import cv2

import numpy as np

import torch

from detector.ssd.utils.misc import Timer
from detector.ssd.mobilenetv3_ssd_lite import (
	create_mobilenetv3_large_ssd_lite,
	create_mobilenetv3_small_ssd_lite,
	create_mobilenetv3_ssd_lite_predictor
)

from dataset.voc import VOCDetection

from detector.ssd.utils import box_utils
from detector.ssd.utils.misc import Timer

from util import measurements

from storage.util import load


def group_annotation_by_class(dataset):
	true_case_stat = {}
	all_gt_boxes = {}
	all_difficult_cases = {}

	for i in range(len(dataset)):
		image_id, annotation = dataset.get_annotation(i)
		gt_boxes, classes, is_difficult = annotation
		gt_boxes = torch.tensor(gt_boxes)

		for i, difficult in enumerate(is_difficult):
			class_index = int(classes[i])
			gt_box = gt_boxes[i]
			if not difficult:
				true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

			if class_index not in all_gt_boxes:
				all_gt_boxes[class_index] = {}
			if image_id not in all_gt_boxes[class_index]:
				all_gt_boxes[class_index][image_id] = []
			all_gt_boxes[class_index][image_id].append(gt_box)

			if class_index not in all_difficult_cases:
				all_difficult_cases[class_index]={}
			if image_id not in all_difficult_cases[class_index]:
				all_difficult_cases[class_index][image_id] = []
			all_difficult_cases[class_index][image_id].append(difficult)

	for class_index in all_gt_boxes:
		for image_id in all_gt_boxes[class_index]:
			all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])

	return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
										preds, iou_threshold, use_2007_metric):
	image_ids = []
	boxes = []
	scores = []

	for line in preds:
		image_ids.append(line[0])
		box = line[1].unsqueeze(0)
		boxes.append(box)
		scores.append(line[2].item())

	scores = np.array(scores)
	sorted_indexes = np.argsort(-scores)
	boxes = [boxes[i] for i in sorted_indexes]
	image_ids = [image_ids[i] for i in sorted_indexes]
	true_positive = np.zeros(len(image_ids))
	false_positive = np.zeros(len(image_ids))
	matched = set()

	for i, image_id in enumerate(image_ids):
		box = boxes[i]
		if image_id not in gt_boxes:
			false_positive[i] = 1
			continue

		gt_box = gt_boxes[image_id]
		ious = box_utils.iou_of(box, gt_box)
		max_iou = torch.max(ious).item()
		max_arg = torch.argmax(ious).item()

		if max_iou > iou_threshold:
			if difficult_cases[image_id][max_arg] == 0:
				if (image_id, max_arg) not in matched:
					true_positive[i] = 1
					matched.add((image_id, max_arg))
				else:
					false_positive[i] = 1
		else:
			false_positive[i] = 1

	true_positive = true_positive.cumsum()
	false_positive = false_positive.cumsum()

	precision = true_positive / (true_positive + false_positive)
	recall = true_positive / num_true_cases

	if use_2007_metric:
		return measurements.compute_voc2007_average_precision(precision, recall)
	else:
		return measurements.compute_average_precision(precision, recall)


def main():
	parser = argparse.ArgumentParser(
		description="Calculate Pascal VOC evaluation metrics")

	parser.add_argument("--model-path", '-p', type=str, required=True,
						help="path to the trained model")

	parser.add_argument("--dataset", type=str,
						help="dataset directory path")

	parser.add_argument("--nms_method", type=str, default="hard")

	parser.add_argument("--iou_threshold", type=float, default=0.5)

	args = parser.parse_args()

	timer = Timer()

	class_names = ('BACKGROUND',
			'aeroplane', 'bicycle', 'bird', 'boat',
			'bottle', 'bus', 'car', 'cat', 'chair',
			'cow', 'diningtable', 'dog', 'horse',
			'motorbike', 'person', 'pottedplant',
			'sheep', 'sofa', 'train', 'tvmonitor')

	dataset = VOCDetection(root=args.dataset, year='2007', image_set='val')

	model, class_names = load(args.model_path)
	model = model.to("cpu")
	model.eval()

	if dataset.class_names != class_names:
		print("Dataset classes don't match the classes "
			  "the specified model is trained with. "
			  "No chance to get valid results, so I give up.")
		sys.exit(-1)

	predictor = create_mobilenetv3_ssd_lite_predictor(
		model, nms_method=args.nms_method)

	true_case_stat, all_gb_boxes, all_difficult_cases = \
		group_annotation_by_class(dataset)

	results_per_class = dict()
	for i in range(len(dataset)):
		image = dataset.get_image(i)
		image_id = dataset.ids[i]
		boxes, labels, probs = predictor.predict(image)

		for box, label, prob in zip(boxes, labels, probs):
			if label.item() not in results_per_class:
				results_per_class.update({ label.item(): [] })

			results_per_class[label.item()].append((image_id, box, prob))

	aps = []
	print("\n\nAverage Precision Per-class:")
	for class_index, class_name in enumerate(class_names):
		if class_index == 0:
			continue

		ap = compute_average_precision_per_class(
			true_case_stat[class_index],
			all_gb_boxes[class_index],
			all_difficult_cases[class_index],
			results_per_class[class_index],
			args.iou_threshold,
			use_2007_metric=False
		)
		aps.append(ap)

		print(f"{class_name}: {ap}")

	print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")


if __name__ == "__main__":
	try:
		main()

	except KeyboardInterrupt:
		sys.exit(0)
