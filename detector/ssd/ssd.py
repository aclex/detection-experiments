import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from detector.ssd.utils import box_utils

from backbone.mobilenetv3 import Block, hswish

from nn.separable_conv_2d import SeparableConv2d


class SSD(nn.Module):
    def __init__(self, num_classes, backbone, arch_name, config=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.backbone = backbone
        self.arch_name = arch_name

        self.extras = nn.ModuleList([
            Block(3, self.backbone.out_channels[-1], 256, 512,
                  hswish(), None, stride=2),
            Block(3, 512, 128, 256, hswish(), None, stride=2),
            Block(3, 256, 128, 256, hswish(), None, stride=2),
            Block(3, 256, 64, 64, hswish(), None, stride=2)
        ])

        self.classification_headers = nn.ModuleList([
            SeparableConv2d(in_channels=self.backbone.out_channels[-2],
                            out_channels=6 * num_classes,
                            kernel_size=3, padding=1),
            SeparableConv2d(in_channels=self.backbone.out_channels[-1],
                            out_channels=6 * num_classes,
                            kernel_size=3, padding=1),
            SeparableConv2d(in_channels=512, out_channels=6 * num_classes,
                            kernel_size=3, padding=1),
            SeparableConv2d(in_channels=256, out_channels=6 * num_classes,
                            kernel_size=3, padding=1),
            SeparableConv2d(in_channels=256, out_channels=6 * num_classes,
                            kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ])

        self.regression_headers = nn.ModuleList([
            SeparableConv2d(in_channels=self.backbone.out_channels[-2],
                            out_channels=6 * 4,
                            kernel_size=3, padding=1, onnx_compatible=False),
            SeparableConv2d(in_channels=self.backbone.out_channels[-1],
                            out_channels=6 * 4, kernel_size=3,
                            padding=1, onnx_compatible=False),
            SeparableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3,
                            padding=1, onnx_compatible=False),
            SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3,
                            padding=1, onnx_compatible=False),
            SeparableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3,
                            padding=1, onnx_compatible=False),
            nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ])

        self.config = config

    def forward(self, x):
        confidences = []
        locations = []

        cs = self.backbone.forward(x)

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

    def load_backbone_weights(self, path):
        self.backbone.load_state_dict(
            torch.load(path, map_location=lambda storage, loc: storage),
            strict=True)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
