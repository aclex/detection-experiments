import torch

from backbone.rw_mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small

from detector.ssd.ssd import SSD
from detector.ssd.predictor import Predictor
import detector.ssd.config as config


def create_mobilenetv3_large_ssd_lite(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, pretrained=False):
    base_net = MobileNetV3_Large(pretrained=pretrained)

    return SSD(num_classes, base_net, "mb3-large-ssd-lite", config=config)


def create_mobilenetv3_small_ssd_lite(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, pretrained=False):
    base_net = MobileNetV3_Small(pretrained=pretrained)

    return SSD(num_classes, base_net, "mb3-small-ssd-lite", config=config)


def create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor


name_to_ctor = {
    "mb3-small-ssd-lite": create_mobilenetv3_small_ssd_lite,
    "mb3-large-ssd-lite": create_mobilenetv3_large_ssd_lite
}
