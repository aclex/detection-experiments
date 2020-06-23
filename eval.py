import sys
import argparse
import pathlib
import logging

import torch

from detector.ssd.utils.misc import Timer
from detector.ssd.mobilenetv3_ssd_lite import (
	create_mobilenetv3_large_ssd_lite,
	create_mobilenetv3_small_ssd_lite,
	create_mobilenetv3_ssd_lite_predictor
)

from dataset.voc import VOCDetection
from dataset.coco import CocoDetection

from storage.util import load

from metric import pascal_voc, coco


def main():
	parser = argparse.ArgumentParser(
		description="Calculate Pascal VOC evaluation metrics")

	parser.add_argument("--model-path", '-p', type=str, required=True,
	                    help="path to the trained model")

	parser.add_argument('--dataset-style', type=str, required=True,
	                    help="style of dataset "
	                    "(supported are 'pascal-voc' and 'coco')")

	parser.add_argument('--image-set', type=str, default="test",
	                    help='image set (annotation file basename for COCO) '
	                    'to use for evaluation')

	parser.add_argument("--dataset", type=str, help="dataset directory path")

	parser.add_argument("--metric", '-m', type=str, default='pascal-voc',
	                    help="metric to calculate ('pascal-voc' or 'coco')")

	parser.add_argument("--nms-method", type=str, default="hard")

	parser.add_argument("--iou-threshold", type=float, default=0.5,
	                    help="IOU threshold (for Pascal VOC metric)")

	parser.add_argument("--metric-score-threshold", type=float, default=0.5,
	                    help="Score threshold (for calculating TP, FP, FN "
	                    "metrics along with Pascal VOC metric)")

	parser.add_argument("--use-2007", action='store_true',
	                    help="Use 2007 calculation algorithm "
	                    "(for Pascal VOC metric)")

	parser.add_argument('--device', type=str, help='device to use')

	args = parser.parse_args()

	if args.device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		device = args.device

	if device.startswith("cuda"):
		logging.info("Use CUDA")

	timer = Timer()

	if args.dataset_style == 'pascal-voc':
		dataset = VOCDetection(root=args.dataset,
		                       image_set=args.image_set)

	elif args.dataset_style == 'coco':
		dataset = CocoDetection(root=args.dataset,
		                        ann_file="%s.json" % args.image_set)


	model, class_names = load(args.model_path, device=device,
	                          inference=True)
	model.eval()

	if dataset.class_names != class_names:
		print("Dataset classes don't match the classes "
		      "the specified model is trained with. "
		      "No chance to get valid results, so I give up.")
		sys.exit(-1)

	predictor = create_mobilenetv3_ssd_lite_predictor(
		model, nms_method=args.nms_method, device=device)

	if args.metric == 'pascal-voc':
		logging.info("Calculating Pascal VOC metric...")
		pascal_voc.eval(dataset, predictor,
		                iou_threshold=args.iou_threshold,
		                metric_score_threshold=args.metric_score_threshold,
		                use_2007_metric=args.use_2007)

	elif args.metric == 'coco':
		logging.info("Calculating COCO metric...")
		coco.eval(dataset, predictor)

	else:
		print("Metric %s is not supported" % args.metric)
		sys.exit(-2)


if __name__ == "__main__":
	try:
		main()

	except KeyboardInterrupt:
		sys.exit(0)
