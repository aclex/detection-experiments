import os
import sys
import argparse
import logging
import itertools

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR

from detector.ssd.utils.misc import Timer

from dataset.loader import load as load_dataset
from dataset.loader import bbox_format as dataset_bbox_format
from dataset.stat import mean_std

from transform.collate import collate

from arch.bootstrap import get_arch

import processing.train
import processing.test

from storage.util import save, load

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
		description='Detection model training utility')

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

	parser.add_argument(
		'--continue-training', '-p',
		help='continue training session for the previously trained model at '
		'the specified path')
	parser.add_argument(
		'--last-epoch', default=-1, type=int,
		help='last epoch to continue training session at (default is -1)')


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

	if args.continue_training is not None:
		logging.info("Loading network")
		arch, net, class_names = load(
			args.continue_training, device=device)
	else:
		arch = get_arch(args.net_config)

	bbox_format = dataset_bbox_format(args.dataset_style)

	train_mean, train_std = mean_std(
		args.dataset_style,
		args.dataset,
		args.train_image_set)

	train_transform = processing.train.Pipeline(
		[arch.image_size] * 2,
		train_mean, train_std,
		bbox_format=bbox_format)

	if args.val_dataset is not None:
		val_dataset_root = args.val_dataset
	else:
		val_dataset_root = args.dataset

	val_mean, val_std = mean_std(
		args.dataset_style,
		val_dataset_root,
		args.val_image_set)

	val_transform = processing.test.Pipeline(
		[arch.image_size] * 2,
		val_mean, val_std,
		bbox_format=bbox_format)

	logging.info("Loading datasets...")

	dataset = load_dataset(
			args.dataset_style,
			args.dataset,
			args.train_image_set,
			train_transform)

	num_classes = len(dataset.class_names)

	logging.info("Train dataset size: {}".format(len(dataset)))

	# don't allow the last batch be of length 1
	# to not lead our dear BatchNorms to crash on that
	drop_last = len(dataset) % args.batch_size > 0

	train_loader = DataLoader(
		dataset, args.batch_size, collate_fn=collate,
		num_workers=args.num_workers,
		shuffle=True, drop_last=drop_last)

	val_dataset = load_dataset(
			args.dataset_style,
			val_dataset_root,
			args.val_image_set,
			val_transform)

	logging.info("Validation dataset size: {}".format(len(val_dataset)))

	val_loader = DataLoader(
		val_dataset, args.batch_size, collate_fn=collate,
		num_workers=args.num_workers,
		shuffle=False, drop_last=drop_last)

	if args.continue_training is None:
		logging.info("Building network")
		backbone_pretrained = args.backbone_pretrained is not None
		net = arch.build(num_classes, backbone_pretrained, args.batch_size)

		if backbone_pretrained and args.backbone_weights is not None:
			logging.info(f"Load backbone weights from {args.backbone_weights}")
			timer.start("Loading backbone model")
			net.load_backbone_weights(args.backbone_weights)
			logging.info(f'Took {timer.end("Loading backbone model"):.2f}s.')

	if args.freeze_backbone:
		net.freeze_backbone()

	net.to(device)

	last_epoch = args.last_epoch

	criterion = arch.loss(net, device)
	mapper = arch.mapper(net, device)

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

	if args.continue_training is None:
		optim_params = net.parameters()
	else:
		optim_params = [{"params": net.parameters(), "initial_lr": args.lr}]

	optimizer = optim_class(optim_params, **optim_kwargs)
	logging.info(f"Optimizer parameters used: {optim_kwargs}")

	if args.scheduler == 'multi-step':
		logging.info("Uses MultiStepLR scheduler.")
		milestones = [int(v.strip()) for v in args.milestones.split(",")]
		scheduler = MultiStepLR(
			optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch)
	else:
		logging.info("Uses Cosine annealing warm restarts scheduler.")
		# CosineAnnealingWarmRestarts has a bug with `last_epoch` != -1,
		# so we don't set it
		scheduler = CosineAnnealingWarmRestarts(
			optimizer, T_0=args.t0, T_mult=args.t_mult, eta_min=1e-5)

	os.makedirs(args.checkpoint_path, exist_ok=True)

	logging.info(f"Start training from epoch {last_epoch + 1}.")
	for epoch in range(last_epoch + 1, last_epoch + args.num_epochs + 1):
		loop(
			train_loader, net, mapper, criterion,
			optimizer, device=device, epoch=epoch)
		scheduler.step()

		if (epoch > 0 and epoch % args.val_epochs == 0 or
				epoch == args.num_epochs - 1):
			val_loss = loop(
				val_loader, net, mapper, criterion,
				device=device, epoch=epoch)

			filename = f"{arch.name}-Epoch-{epoch}-Loss-{val_loss}.pth"
			model_path = os.path.join(args.checkpoint_path, filename)
			save(arch, net, dataset.class_names, model_path)
			logging.info(f"Saved model {model_path}")


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
