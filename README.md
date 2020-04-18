# PyTorch implementation of interesting object detection models

This project is supposed to include some interesting object detection architectures implemented using PyTorch framework. The main priorities are:
* pure PyTorch implementation, without using additional frameworks on top of that and as little wrappers and dependencies, as possible
* modern interesting architectures with competitive performance, modern optimizers, augmentation and learning strategies
* CPU and GPU support to allow one to try things as quickly, as possible, even without modern GPU-card installed and properly configured
* easy inference scenarios, especially on 'edge' devices
* clean code ready to use as kind of tutorial of what really happens inside

Implementations are mostly based on:
- backbones by Ross Wightman ([pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch))
- augmentations from [albumentations](https://github.com/albumentations-team/albumentations)
- modern optimizers from [Best Deep Learning Optimizers](https://github.com/lessw2020/Best-Deep-Learning-Optimizers)

## MobileNetV3 + SSDLite

Hugely inspired by [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) project, contains MobileNetV3 backbone taken from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by Ross Wightman (alternative backbone is taken from https://github.com/xiaolai-sqlai/mobilenetv3, it seems to deviate from the paper in some parts, but provides more control on the model structure, should you need to modify it manually), with [SSDLite](https://stackoverflow.com/questions/50674448/what-is-the-different-between-ssd-and-ssd-lite-tensorflow) head, as described in [MobileNetV2](https://arxiv.org/abs/1801.04381) and [MobileNetV3](https://arxiv.org/abs/1905.02244) papers.

## Basic operations
### Training

Use `train.py` script for training, with various options to choose dataset type (datasets in [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [MS COCO](http://cocodataset.org) formats supported), optimizer ([Ranger](https://arxiv.org/abs/2004.01461) is the default, [DiffGrad](https://arxiv.org/abs/1909.11015v2) and [AdamW](https://arxiv.org/abs/1711.05101) are the options, SGD with momentum to be back for the sake of comparison), and learning rate scheduling strategy ([Cosine Annealing with Warm Restarts](https://arxiv.org/abs/1608.03983) is the default, multi-step is a baseline option). Example:
```bash
python train.py --dataset-style=coco --dataset /mnt/dataset/vision/COCO --train-image-set=train2012 --val-image-set=val2012 --net mb3-small-ssd-lite --scheduler cosine-wr --lr 0.01 --t0 5 --num-epochs 5
```

You can manually specify the device to use with `--device` key, otherwise CUDA/ROCm device would be used automatically if available or CPU as a fallback. For the case of MobileNetV3-Small-based model one epoch with batch size 32 takes about 5 minutes on CPU (Inter Core i7-6700K) and 1 minute on GPU (AMD Radeon VII) to train, with first sane predictions being available after 10-20 epochs and convergence after 150-200 epochs.

If you would like to use the backbone pre-trained on [ImageNet](http://www.image-net.org/) (not currently available for the used backbone variants), please also specify `--pretrained-backbone` key. Weights would be downloaded automatically, if available. Though for MobileNet-based models this doesn't appear to present any particular sense.

### Inference

To run a model on an image:
```bash
python run.py --image --model-path output/model.pth sample.jpg -o processed.jpg
```
or a video:
```bash
python run.py --video --model-path output/model.pth sample.mkv -o processed.mkv
```

For inference scenarios on [NVIDIA Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) please refer to the additional instructions in [jetson](jetson/README.md) directory.

### Evaluation

To collect the model performance statistics per Pascal VOC or COCO metrics (`pascal-voc` and `coco`, correspondingly, selected with `-m/--metric` key) please use the `eval.py` script:
```bash
python eval.py -m pascal-voc --dataset-style coco --dataset /path/to/the/dataset --image-set test
```

### Dataset visualization

If you want to inspect the dataset samples visually, you can use `visualize.py` script the following way:
```bash
python visualize.py --dataset-style=coco --dataset=/path/to/the/dataset
```
Press any key to switch the samples and `q` to exit.
