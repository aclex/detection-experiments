import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn

from backbone.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small, Block, hswish

from detector.ssd.ssd import SSD, SeparableConv2d
from detector.ssd.predictor import Predictor
import detector.ssd.config as config


name_to_ctor = {
    "mb3-small-ssd-lite": create_mobilenetv3_small_ssd_lite
}


def create_mobilenetv3_large_ssd_lite(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    base_net = MobileNetV3_Large().features

    source_layer_indexes = [ 15, 21 ]
    extras = ModuleList([
        Block(3, 960, 256, 512, hswish(), None, stride=2),
        Block(3, 512, 128, 256, hswish(), None, stride=2),
        Block(3, 256, 128, 256, hswish(), None, stride=2),
        Block(3, 256, 64, 64, hswish(), None, stride=2)
    ])

    regression_headers = ModuleList([
        SeparableConv2d(in_channels=round(112 * width_mult), out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeparableConv2d(in_channels=960, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeparableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeparableConv2d(in_channels=round(112 * width_mult), out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=960, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeparableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobilenetv3_small_ssd_lite(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False):
    base_net = MobileNetV3_Small()

    return SSD(num_classes, base_net, "mb3-small-ssd-lite", config=config)


def create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
