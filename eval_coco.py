import sys
import argparse

import torch

from detector.ssd.mobilenetv3_ssd_lite import (
	create_mobilenetv3_large_ssd_lite,
	create_mobilenetv3_small_ssd_lite,
	create_mobilenetv3_ssd_lite_predictor
)

from dataset.voc import VOCDetection

from storage.util import load

from metric import coco


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

	coco.eval(dataset, predictor)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
