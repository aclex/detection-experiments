import os
import sys
import argparse
import logging

import torch

import albumentations as A

from dataset.voc import VOCDetection
from dataset.coco import CocoDetection
from dataset.stat import get_mean_std

from transform.to_tensor import ToTensor


def main():
	parser = argparse.ArgumentParser(
		description='Dataset statistics counting utility')

	parser.add_argument(
		'--dataset-style', type=str, required=True,
		help="style of dataset (supported are 'pascal-voc' and 'coco')")
	parser.add_argument('--dataset', required=True, help='dataset path')
	parser.add_argument(
		'--image-set', type=str, default="train",
		help='image set (annotation file basename for COCO) '
		'to use for calculation')

	pipeline = A.Compose([
		A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
		ToTensor()
	])

	args = parser.parse_args()

	if args.dataset_style == 'pascal-voc':
		dataset = VOCDetection(
			root=args.dataset,
			image_set=args.image_set,
			transform=train_transform)
	elif args.dataset_style == 'coco':
		dataset = CocoDetection(
			root=args.dataset,
			ann_file="%s.json" % args.image_set,
			transform=pipeline)
	else:
		print("Dataset style %s is not supported" % args.dataset_style)
		sys.exit(-1)

	print("Dataset size: {}".format(len(dataset)))

	mean, std = get_mean_std(dataset)

	print("Dataset mean: {}, std: {}".format(mean, std))


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
