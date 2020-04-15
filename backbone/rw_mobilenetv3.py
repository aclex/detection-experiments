import functools

import torch

import backbone.outlet.pytorch_image_models.timm as outlet


MobileNetV3_Large = functools.partial(outlet.models.mobilenetv3_large_100,
                                      features_only=True,
                                      out_indices=(2, 3),
                                      feature_location="bottleneck")

MobileNetV3_Small = functools.partial(outlet.models.mobilenetv3_small_100,
                                      features_only=True,
                                      out_indices=(2, 3),
                                      feature_location="bottleneck")
