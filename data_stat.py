import os
import sys
import argparse
import logging

import torch

import albumentations as A

from dataset.voc import VOCDetection
from dataset.coco import CocoDetection
from dataset.stat import mean_std

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

	mean, std = mean_std(args.dataset_style, args.dataset, args.image_set)

	print("Dataset mean: {}, std: {}".format(mean, std))


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
