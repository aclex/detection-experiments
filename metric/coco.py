from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from transform.convert_bbox_format import BboxFormatConvert

from util.progress import interactive


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


def eval(dataset, predictor):
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

	for i, c in enumerate(dataset.class_names):
		gt_coco["categories"].append(create_coco_category(i, c))

	dt_coco = {
		"licenses": gt_coco["licenses"],
		"annotations": [],
		"categories": gt_coco["categories"]
	}

	input_bbox_converter = BboxFormatConvert(
		source_format=dataset.bbox_format, target_format='coco')

	output_bbox_converter = BboxFormatConvert(
		source_format='pascal_voc', target_format='coco')

	for i in interactive(range(len(dataset))):
		sample = dataset[i]
		image = sample['image']
		height, width = image.shape[:2]

		image_record = create_coco_image_record(i, (width, height))
		gt_coco["images"].append(image_record)

		coco_sample = input_bbox_converter(**sample)
		boxes = coco_sample['bboxes']
		labels = coco_sample['category_id']
		scores = [1 for _ in labels]
		gt_anns = create_coco_annotations(i, boxes, labels, scores)
		gt_coco["annotations"].extend(gt_anns)

		boxes, labels, probs = predictor.predict(image, prob_threshold=0)
		boxes = [b.tolist() for b in boxes]
		labels = labels.tolist()
		probs = probs.tolist()
		boxes = output_bbox_converter(image=image, bboxes=boxes)["bboxes"]

		dt_anns = create_coco_annotations(i, boxes, labels, probs)
		dt_coco["annotations"].extend(dt_anns)

	dt_coco.update({ "images": gt_coco["images"] })

	gt_coco_obj = COCO()
	gt_coco_obj.dataset = gt_coco
	gt_coco_obj.createIndex()
	dt_coco_obj = COCO()
	dt_coco_obj.dataset = dt_coco
	dt_coco_obj.createIndex()

	e = COCOeval(gt_coco_obj, dt_coco_obj, iouType='bbox')

	e.evaluate()
	e.accumulate()

	e.summarize()

	result = [
		{
			"iou": [e.params.iouThrs[0], e.params.iouThrs[-1]],
			"area": [e.params.areaRng[0], e.params.areaRng[-1]],
			"max_dets": e.params.maxDets[2],
			"ap": e.stats[0]
		},
		{
			"iou": 0.5,
			"area": [e.params.areaRng[0], e.params.areaRng[-1]],
			"max_dets": e.params.maxDets[2],
			"ap": e.stats[1]
		},
		{
			"iou": 0.75,
			"area": [e.params.areaRng[0], e.params.areaRng[-1]],
			"max_dets": e.params.maxDets[2],
			"ap": e.stats[2]
		},
		{
			"iou": [e.params.iouThrs[0], e.params.iouThrs[-1]],
			"area": e.params.areaRng[1],
			"max_dets": e.params.maxDets[2],
			"ap": e.stats[3]
		},
		{
			"iou": [e.params.iouThrs[0], e.params.iouThrs[-1]],
			"area": e.params.areaRng[2],
			"max_dets": e.params.maxDets[2],
			"ap": e.stats[4]
		},
		{
			"iou": [e.params.iouThrs[0], e.params.iouThrs[-1]],
			"area": e.params.areaRng[3],
			"max_dets": e.params.maxDets[2],
			"ap": e.stats[5]
		},
		{
			"iou": [e.params.iouThrs[0], e.params.iouThrs[-1]],
			"area": [e.params.areaRng[0], e.params.areaRng[-1]],
			"max_dets": e.params.maxDets[0],
			"ar": e.stats[6]
		},
		{
			"iou": [e.params.iouThrs[0], e.params.iouThrs[-1]],
			"area": [e.params.areaRng[0], e.params.areaRng[-1]],
			"max_dets": e.params.maxDets[1],
			"ar": e.stats[7]
		},
		{
			"iou": [e.params.iouThrs[0], e.params.iouThrs[-1]],
			"area": [e.params.areaRng[0], e.params.areaRng[-1]],
			"max_dets": e.params.maxDets[2],
			"ar": e.stats[8]
		},
		{
			"iou": [e.params.iouThrs[0], e.params.iouThrs[-1]],
			"area": e.params.areaRng[1],
			"max_dets": e.params.maxDets[2],
			"ar": e.stats[9]
		},
		{
			"iou": [e.params.iouThrs[0], e.params.iouThrs[-1]],
			"area": e.params.areaRng[2],
			"max_dets": e.params.maxDets[1],
			"ar": e.stats[10]
		},
		{
			"iou": [e.params.iouThrs[0], e.params.iouThrs[-1]],
			"area": e.params.areaRng[3],
			"max_dets": e.params.maxDets[2],
			"ar": e.stats[11]
		}
	]

	return result
