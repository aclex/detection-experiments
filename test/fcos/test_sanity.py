import pytest

import torch

from backbone.rw_mobilenetv3 import MobileNetV3_Small
from nn.separable_conv_2d import SeparableConv2d
from detector.fcos.model import Blueprint


@pytest.fixture
def model():
	b = MobileNetV3_Small(pretrained=False)
	m = Blueprint(
		"test", b, 2,
		num_channels=48, num_levels=3, num_fpn_layers=1, num_blocks=2,
		conv=SeparableConv2d)
	m.class_names = ["background", "person"]

	return m


def test_sanity(model):
	a = torch.ones((2, 3, 512, 512))

	r = model.forward(a)

	assert len(r) == model.head.num_levels


def test_onnx_export(model):
	a = torch.ones((1, 3, 256, 256)).to(dtype=torch.float32)

	model.to(dtype=torch.float32)

	model.eval()

	torch.onnx.export(
		model, a, "test.onnx",
		input_names=["img"],
		output_names=["p1", "p2", "p3"],
		opset_version=11,
		do_constant_folding=True,
		keep_initializers_as_inputs=True)
