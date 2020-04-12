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

from detector.ssd.utils.misc import Timer

from storage.util import load

from metric import pascal_voc


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

	pascal_voc.eval(dataset, predictor, args.iou_threshold, False)


if __name__ == "__main__":
	try:
		main()

	except KeyboardInterrupt:
		sys.exit(0)
