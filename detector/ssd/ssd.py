import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from detector.ssd.utils import box_utils

from backbone.mobilenetv3 import Block, hswish

from nn.separable_conv_2d import SeparableConv2d


class SSD(nn.Module):
    def __init__(self, num_classes, base_net, arch_name, config=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.arch_name = arch_name

        self.extras = nn.ModuleList([
            Block(3, 576, 256, 512, hswish(), None, stride=2),
            Block(3, 512, 128, 256, hswish(), None, stride=2),
            Block(3, 256, 128, 256, hswish(), None, stride=2),
            Block(3, 256, 64, 64, hswish(), None, stride=2)
        ])

        self.classification_headers = nn.ModuleList([
            SeparableConv2d(in_channels=48, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeparableConv2d(in_channels=576, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeparableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeparableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            SeparableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ])

        self.regression_headers = nn.ModuleList([
            SeparableConv2d(in_channels=48, out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            SeparableConv2d(in_channels=576, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeparableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
            nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ])

        self.config = config

    def forward(self, x):
        confidences = []
        locations = []

        cs = self.base_net.forward(x)

        for i, c in enumerate(cs):
            confidence, location = self.compute_header(i, c)
            x = c
            confidences.append(confidence)
            locations.append(location)

        header_index = i + 1

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        return confidences, locations

    def get_predictions(self, output):
        confidences, locations = output

        confidences = F.softmax(confidences, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            locations, self.config.priors,
            self.config.center_variance, self.config.size_variance
        )
        boxes = box_utils.center_form_to_corner_form(boxes)

        return confidences, boxes

    def predict(self, x):
        output = self.forward(x)
        return self.get_predictions(output)

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
