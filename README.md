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

## Backbone architectures

The project was initially started on top of [custom implementation](https://github.com/xiaolai-sqlai/mobilenetv3) of [MobileNetV3](https://arxiv.org/abs/1905.02244), mainly for educational purposes, but then adapted to use some of those nice and modern implementations available in [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by Ross Wightman. Please, note, that not all of the architectures (which can, by the way, be listed with `timm.list_models()`) may be used as a backbone in this project. However, many of them do.

Apart from that, several interesting backbone architetures were also added, namely [GhostNet](https://arxiv.org/abs/1911.11907) and [TinyNet](https://arxiv.org/abs/2010.14819) from [CV-Backbones](https://github.com/huawei-noah/CV-Backbones) by Huawei. 

## FPN

In recent presets, BiFPN from [EfficientDet](https://arxiv.org/abs/1911.09070) is used in own implementation. SSDLite head still contains its own FPN implementation.

## Heads

### SSDLite

Hugely inspired by [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) project with [SSDLite](https://stackoverflow.com/questions/50674448/what-is-the-different-between-ssd-and-ssd-lite-tensorflow) head implementation. It's quite simple and fast, yet so rigid form serious tweaking and anchor-based.

### FCOS

Custom and flexible implementation mainly inspired by popular anchor-free architecture of [FCOS](https://arxiv.org/abs/2006.09214), still not exactly replica of [original implementation](https://github.com/tianzhi0549/FCOS).

### FoveaBox

Does the centerness branch of FCOS make any sense, after all? :) Another custom implementation,
now inspired by anchor-free [FoveaBox](https://arxiv.org/abs/1904.03797) architecture, was added to answer this strange question. Shares much code with FCOS head.


# Presets

To maintain the balance between code sanity and readability and configuration flexibility,
the preset concept is adopted. Following this idea, there are several 'factories' in the
code, ready to build certain models, mainly dictated by the head architecture. Details of
model structure (e.g. backbone or FPN architecture used, values of meta-parameters, image resolution etc.) are defined in small JSON configuration files called presets. Some common versions are placed in `preset` subdirectory.

## Basic operations
### Training

Use `train.py` script for training, with various options to choose dataset type (datasets in [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [MS COCO](http://cocodataset.org) formats supported), optimizer ([Ranger](https://arxiv.org/abs/2004.01461) is the default, [DiffGrad](https://arxiv.org/abs/1909.11015v2) and [AdamW](https://arxiv.org/abs/1711.05101) are the options, SGD with momentum to be back for the sake of comparison), and learning rate scheduling strategy ([Cosine Annealing with Warm Restarts](https://arxiv.org/abs/1608.03983) is the default, multi-step is a baseline option). Example:
```bash
python train.py --dataset-style=coco --dataset /mnt/dataset/vision/COCO --train-image-set=train2012 --val-image-set=val2012 --net-config preset/tinynet-a-bifpn-foveabox.json --scheduler cosine-wr --lr 0.01 --t0 5 --num-epochs 5
```

You can manually specify the device to use with `--device` key, otherwise CUDA/ROCm device would be used automatically if available or CPU as a fallback. For the case of MobileNetV3-Small-based model one epoch with batch size 32 takes about 5 minutes on CPU (Inter Core i7-6700K) and 1 minute on GPU (AMD Radeon VII) to train, with first sane predictions being available after 10-20 epochs and convergence after 128-256+ epochs, depending on the dataset size.

If you would like to use the backbone pre-trained on [ImageNet](http://www.image-net.org/) (not currently available for the used backbone variants), please also specify `--pretrained-backbone` key. Weights would be downloaded automatically, if available. Though for MobileNet-based models this doesn't appear to present any particular sense.

You can also continue training previously saved model by specifying the path to it with `--continue-training` or `-p` key. It will continue training the model from the epoch 0 of the schedule, which is particularly important when complex strategies like cosine annealing are used. You can define the epoch number to start with `--last-epoch` key. When re-training the existing model on some different
dataset, it is generally advisable to consider decreasing the learning rate or making small warming-up training with minimal learning rate for a few epochs, to prevent gradient explosion.

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
