import numpy as np

import torch

from detector.ssd.utils import box_utils

from util.progress import interactive

from transform.convert_bbox_format import BboxFormatConvert


def compute_average_precision(precision, recall):
	"""
	It computes average precision based on the definition of Pascal Competition. It computes the under curve area
	of precision and recall. Recall follows the normal definition. Precision is a variant.
	pascal_precision[i] = typical_precision[i:].max()
	"""
	# identical but faster version of new_precision[i] = old_precision[i:].max()
	precision = np.concatenate([[0.0], precision, [0.0]])
	for i in range(len(precision) - 1, 0, -1):
		precision[i - 1] = np.maximum(precision[i - 1], precision[i])

	# find the index where the value changes
	recall = np.concatenate([[0.0], recall, [1.0]])
	changing_points = np.where(recall[1:] != recall[:-1])[0]

	# compute under curve area
	areas = (recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]
	return areas.sum()


def compute_voc2007_average_precision(precision, recall):
	ap = 0.
	for t in np.arange(0., 1.1, 0.1):
		if np.sum(recall >= t) == 0:
			p = 0
		else:
			p = np.max(precision[recall >= t])
		ap = ap + p / 11.
	return ap


def group_annotation_by_class(dataset):
	true_case_stat = {}
	all_gt_boxes = {}
	all_difficult_cases = {}

	input_bbox_converter = BboxFormatConvert(
		source_format=dataset.bbox_format, target_format='pascal_voc')

	print("Processing dataset...")
	for i in interactive(range(len(dataset))):
		image_id, annotation = dataset.get_annotation(i)
		gt_boxes, classes = annotation[:2]
		if len(annotation) > 2:
			is_difficult = annotation[2]
		else:
			is_difficult = [False] * len(classes)

		gt_boxes = input_bbox_converter(
			image=dataset.get_image(i), bboxes=gt_boxes)["bboxes"]
		gt_boxes = torch.tensor(gt_boxes)

		for i, difficult in enumerate(is_difficult):
			class_index = int(classes[i])
			gt_box = gt_boxes[i]
			if not difficult:
				true_case_stat[class_index] = \
					true_case_stat.get(class_index, 0) + 1

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


def compute_average_precision_per_class(num_true_cases, gt_boxes,
                                        difficult_cases, preds,
                                        iou_threshold, use_2007_metric):
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
	scores = [scores[i] for i in sorted_indexes]
	image_ids = [image_ids[i] for i in sorted_indexes]
	true_positive = np.zeros(len(image_ids))
	false_positive = np.zeros(len(image_ids))
	matched = set()

	for i, image_id in enumerate(image_ids):
		box = boxes[i]
		if image_id not in gt_boxes:
			false_positive[i] = 1
			continue

		image_gt_boxes = gt_boxes[image_id]
		ious = box_utils.iou_of(box, image_gt_boxes)
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
		return compute_voc2007_average_precision(precision, recall)
	else:
		return compute_average_precision(precision, recall)


def eval(dataset, predictor, iou_threshold=0.5, use_2007_metric=False):
	true_case_stat, all_gt_boxes, all_difficult_cases = \
		group_annotation_by_class(dataset)

	results_per_class = dict()
	for i in interactive(range(len(dataset))):
		image = dataset.get_image(i)
		image_id = dataset.ids[i]
		boxes, labels, probs = predictor.predict(
			image, prob_threshold=0)

		for box, label, prob in zip(boxes, labels, probs):
			if label.item() not in results_per_class:
				results_per_class.update({ label.item(): [] })

			results_per_class[label.item()].append((image_id, box, prob))

	aps = []
	print("\n\nAverage precision per-class:")
	for class_index, class_name in enumerate(dataset.class_names):
		if class_index == 0:
			continue

		ap = compute_average_precision_per_class(
			true_case_stat[class_index],
			all_gt_boxes[class_index],
			all_difficult_cases[class_index],
			results_per_class[class_index],
			iou_threshold,
			use_2007_metric
		)
		aps.append(ap)

		print(f"{class_name}: {ap}")

	avg_ap = sum(aps) / len(aps)
	print(f"\nAverage Precision Across All Classes:{avg_ap}")

	return {
		"per class": aps,
		"rate": avg_ap
	}
