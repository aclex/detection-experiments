import os
import sys
import argparse
import logging
import itertools

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR

from detector.ssd.utils.misc import Timer

from dataset.voc import VOCDetection
from dataset.coco import CocoDetection

from transform.collate import collate

from arch.bootstrap import get_arch

import processing.train
import processing.test

from storage.util import save

from optim.Ranger.ranger import Ranger
from optim.diffgrad.diffgrad import DiffGrad


torch.multiprocessing.set_sharing_strategy('file_system')


def loop(
		loader, net, mapper, criterion, optimizer=None, device=None, epoch=-1):
	training = optimizer is not None
	net.train(training)

	running_loss = {}
	num = 0

	for i, data in enumerate(loader):
		images = data["image"]
		boxes = data["bboxes"]
		labels = data["category_id"]

		images = images.to(device, dtype=torch.float32)
		boxes = [b.to(device, dtype=torch.float32) for b in boxes]
		labels = [l.to(device, dtype=torch.long) for l in labels]

		num += 1

		if training:
			optimizer.zero_grad()

		target = mapper.forward((boxes, labels))

		with torch.set_grad_enabled(training):
			pred = net.forward(images)
			loss_dict = criterion.forward(pred, target)

		if training:
			assert "total" in loss_dict

			loss_dict["total"].backward()
			optimizer.step()

		for k, v in loss_dict.items():
			if not k in running_loss:
				running_loss.update({ k: 0. })
			running_loss[k] += v.item()

	avg_loss = { k: v / num for k, v in running_loss.items() }

	mode_name = "training" if training else "validation"
	info = f"Epoch: {epoch}, Step: {i}, Mode: {mode_name}, Average Loss: "

	for i, (k, v) in enumerate(avg_loss.items()):
		info += f"{k}: {v:.4f}"
		if i < len(avg_loss) - 1:
			info += ", "

	logging.info(info)

	return avg_loss["total"]


def main():
	parser = argparse.ArgumentParser(
		description='FCOS Detector Training With Pytorch')

	parser.add_argument(
		'--dataset-style', type=str, required=True,
		help="style of dataset (supported are 'pascal-voc' and 'coco')")
	parser.add_argument('--dataset', required=True, help='dataset path')
	parser.add_argument(
		'--train-image-set', type=str, default="train",
		help='image set (annotation file basename for COCO) '
		'to use for training')
	parser.add_argument(
		'--val-image-set', type=str, default="val",
		help='image set (annotation file basename for COCO) '
		'to use for validation')
	parser.add_argument(
		'--val-dataset', default=None,
		help='separate validation dataset directory path')

	parser.add_argument(
		'--net-config',
		help="path to network architecture configuration file "
		"(take a look into 'preset' directory for the reference)")

	# Params for optimizer
	parser.add_argument(
		'--optimizer', default="ranger",
		help="optimizer to use ('sgd', 'diffgrad', 'adamw', or 'ranger')")
	parser.add_argument(
		'--lr', '--learning-rate', default=1e-3, type=float,
		help='initial learning rate')
	parser.add_argument(
		'--momentum', default=0.9, type=float,
		help='optional momentum for SGD optimizer (default is 0.9)')
	parser.add_argument(
		'--weight-decay', default=5e-4, type=float,
		help='optional weight decay (L2 penalty) '
		'for SGD optimizer (default is 5e-4)')

	parser.add_argument('--backbone-pretrained', action='store_true')
	parser.add_argument(
		'--backbone-weights',
		help='pretrained weights for the backbone model')
	parser.add_argument('--freeze-backbone', action='store_true')

	# Scheduler
	parser.add_argument(
		'--scheduler', default="cosine-wr", type=str,
		help="scheduler for SGD. It can one of 'multi-step' and 'cosine-wr'")

	# Params for Scheduler
	parser.add_argument(
		'--milestones', default="70,100", type=str,
		help="milestones for MultiStepLR")
	parser.add_argument(
		'--t0', default=10, type=int,
		help='T_0 value for Cosine Annealing Warm Restarts.')
	parser.add_argument(
		'--t-mult', default=2, type=float,
		help='T_mult value for Cosine Annealing Warm Restarts.')

	# Train params
	parser.add_argument('--batch-size', default=32, type=int, help='batch size')
	parser.add_argument(
		'--num-epochs', default=120, type=int, help='number of epochs to train')
	parser.add_argument(
		'--num-workers', default=4, type=int,
		help='number of workers used in dataloading')
	parser.add_argument(
		'--val-epochs', default=5, type=int,
		help='perform validation every this many epochs')
	parser.add_argument(
		'--device', type=str,
		help='device to use for training')

	parser.add_argument(
		'--checkpoint-path', default='output',
		help='directory for saving checkpoint models')


	logging.basicConfig(
		stream=sys.stdout, level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s')

	args = parser.parse_args()
	logging.info(args)

	if args.device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		device = args.device

	if device.startswith("cuda"):
		logging.info("Use CUDA")

	timer = Timer()

	arch = get_arch(args.net_config)

	if args.dataset_style == 'pascal-voc':
		bbox_format = 'pascal_voc'
	elif args.dataset_style == 'coco':
		bbox_format = 'coco'
	else:
		print("Dataset style %s is not supported" % args.dataset_style)
		sys.exit(-1)

	train_transform = processing.train.Pipeline(
		[arch.image_size] * 2,
		arch.image_mean, arch.image_std,
		bbox_format=bbox_format)

	test_transform = processing.test.Pipeline(
		[arch.image_size] * 2,
		arch.image_mean, arch.image_std,
		bbox_format=bbox_format)

	logging.info("Loading datasets...")

	if args.dataset_style == 'pascal-voc':
		dataset = VOCDetection(
			root=args.dataset,
			image_set=args.train_image_set,
			transform=train_transform)
	elif args.dataset_style == 'coco':
		dataset = CocoDetection(
			root=args.dataset,
			ann_file="%s.json" % args.train_image_set,
			transform=train_transform)

	num_classes = len(dataset.class_names)

	logging.info("Train dataset size: {}".format(len(dataset)))

	# don't allow the last batch be of length 1
	# to not lead our dear BatchNorms to crash on that
	drop_last = len(dataset) % args.batch_size == 1

	train_loader = DataLoader(
		dataset, args.batch_size, collate_fn=collate,
		num_workers=args.num_workers,
		shuffle=True, drop_last=drop_last)

	if args.val_dataset is not None:
		val_dataset_root = args.val_dataset
	else:
		val_dataset_root = args.dataset

	if args.dataset_style == 'pascal-voc':
		val_dataset = VOCDetection(
			root=val_dataset_root,
			image_set=args.val_image_set,
			transform=test_transform)
	elif args.dataset_style == 'coco':
		val_dataset = CocoDetection(
			root=val_dataset_root,
			ann_file="%s.json" % args.val_image_set,
			transform=test_transform)

	logging.info("Validation dataset size: {}".format(len(val_dataset)))

	val_loader = DataLoader(
		val_dataset, args.batch_size, collate_fn=collate,
		num_workers=args.num_workers,
		shuffle=False)

	logging.info("Building network")
	backbone_pretrained = args.backbone_pretrained is not None
	net = arch.build(num_classes, backbone_pretrained)

	if backbone_pretrained and args.backbone_weights is not None:
		logging.info(f"Load backbone weights from {args.backbone_weights}")
		timer.start("Loading backbone model")
		net.load_backbone_weights(args.backbone_weights)
		logging.info(f'Took {timer.end("Loading backbone model"):.2f}s.')

	if args.freeze_backbone:
		net.freeze_backbone()

	net.to(device)

	last_epoch = -1

	criterion = arch.loss(net)

	mapper = arch.mapper(net)

	optim_kwargs = {
		"lr": args.lr,
		"weight_decay": args.weight_decay
	}

	if args.optimizer == "sgd":
		optim_class = torch.optim.SGD
		optim_kwargs.update({
			"momentum": args.momentum
		})
	elif args.optimizer == "adamw":
		optim_class = torch.optim.AdamW
	elif args.optimizer == "diffgrad":
		optim_class = DiffGrad
	else:
		optim_class = Ranger

	optimizer = optim_class(net.parameters(), **optim_kwargs)
	logging.info(f"Optimizer parameters used: {optim_kwargs}")

	if args.scheduler == 'multi-step':
		logging.info("Uses MultiStepLR scheduler.")
		milestones = [int(v.strip()) for v in args.milestones.split(",")]
		scheduler = MultiStepLR(
			optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch)
	else:
		logging.info("Uses Cosine annealing warm restarts scheduler.")
		scheduler = CosineAnnealingWarmRestarts(
			optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=1e-5)

	os.makedirs(args.checkpoint_path, exist_ok=True)

	logging.info(f"Start training from epoch {last_epoch + 1}.")
	for epoch in range(last_epoch + 1, args.num_epochs):
		loop(
			train_loader, net, mapper, criterion,
			optimizer, device=device, epoch=epoch)
		scheduler.step()

		if epoch % args.val_epochs == 0 or epoch == args.num_epochs - 1:
			val_loss = loop(val_loader, net, mapper, criterion, device=device)
			filename = f"{arch.name}-Epoch-{epoch}-Loss-{val_loss}.pth"
			model_path = os.path.join(args.checkpoint_path, filename)
			save(net, dataset.class_names, model_path)
			logging.info(f"Saved model {model_path}")


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
