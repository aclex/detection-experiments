import sys
import argparse

import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detector.ssd.mobilenetv3_ssd_lite import (
	create_mobilenetv3_large_ssd_lite,
	create_mobilenetv3_small_ssd_lite,
	create_mobilenetv3_ssd_lite_predictor
)

from dataset.voc import VOCDetection

from transform.convert_bbox_format import ConvertBboxFormat

from storage.util import load


def create_coco_annotations(image_id, boxes, labels, scores, gt=False):
	if not hasattr(create_coco_annotations, "ann_id"):
		create_coco_annotations.ann_id = 0

	anns = []
	for n, (box, label, score) in enumerate(zip(boxes, labels, scores)):
		ann = {
			"image_id": int(image_id),
			"category_id": int(label),
			"id": create_coco_annotations.ann_id,
			"iscrowd": 0,
			"segmentation": [],
			"area": box[2] * box[3],
			"bbox": list(box),
		}
		if not gt:
			ann.update({ "score": score })
		anns.append(ann)
		create_coco_annotations.ann_id += 1

	return anns


def create_coco_image_record(image_index, image_size):
	return {
		"coco_url": "",
		"date_captured": "",
		"flickr_url": "",
		"license": 0,
		"id": image_index,
		"file_name": "",
		"height": image_size[1],
		"width": image_size[0]
	}


def create_coco_category(category_id, category_name):
	return {
		"id": category_id,
		"name": category_name,
		"supercategory": ""
	}


def main():
	parser = argparse.ArgumentParser(
		description="Calculate Pascal VOC evaluation metrics")

	parser.add_argument("--model-path", '-p', type=str, required=True,
						help="path to the trained model")

	parser.add_argument("--dataset", type=str,
						help="dataset directory path")

	parser.add_argument("--nms_method", type=str, default="hard")

	args = parser.parse_args()

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

	gt_coco = {
		"licenses": {
			"name": "",
			"id": 0,
			"url": ""
		},
		"images": [],
		"annotations": [],
		"categories": []
	}

	for i, c in enumerate(class_names):
		gt_coco["categories"].append(create_coco_category(i, c))

	dt_coco = {
		"licenses": gt_coco["licenses"],
		"annotations": [],
		"categories": gt_coco["categories"]
	}

	bbox_converter = ConvertBboxFormat(source_format='pascal_voc',
									   target_format='coco')

	for i in range(len(dataset)):
		sample = dataset[i]
		image = sample['image']
		height, width = image.shape[:2]

		image_record = create_coco_image_record(i, (width, height))
		gt_coco["images"].append(image_record)

		coco_sample = bbox_converter(**sample)
		boxes = coco_sample['bboxes']
		labels = coco_sample['category_id']
		scores = [1 for _ in labels]
		gt_anns = create_coco_annotations(i, boxes, labels, scores)
		gt_coco["annotations"].extend(gt_anns)

		boxes, labels, probs = predictor.predict(image)
		boxes = [b.tolist() for b in boxes]
		labels = labels.tolist()
		probs = probs.tolist()
		boxes = bbox_converter(image=image, bboxes=boxes)["bboxes"]

		dt_anns = create_coco_annotations(i, boxes, labels, probs)
		dt_coco["annotations"].extend(dt_anns)

	dt_coco.update({ "images": gt_coco["images"] })

	gt_coco_obj = COCO()
	gt_coco_obj.dataset = gt_coco
	gt_coco_obj.createIndex()
	dt_coco_obj = COCO()
	dt_coco_obj.dataset = dt_coco
	dt_coco_obj.createIndex()

	eval = COCOeval(gt_coco_obj, dt_coco_obj, iouType='bbox')

	eval.evaluate()
	eval.accumulate()

	eval.summarize()

	result = {
		"Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]": eval.stats[0],
		"Average Precision  (AP) @[ IoU=0.50	  | area=   all | maxDets=100 ]": eval.stats[1],
		"Average Precision  (AP) @[ IoU=0.75	  | area=   all | maxDets=100 ]": eval.stats[2],
		"Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]": eval.stats[3],
		"Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]": eval.stats[4],
		"Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]": eval.stats[5],
		"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]": eval.stats[6],
		"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]": eval.stats[7],
		"Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]": eval.stats[8],
		"Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]": eval.stats[9],
		"Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]": eval.stats[10],
		"Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]": eval.stats[11]
	}

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
