import pytest

import torch

from detector.fcos.model import MobileNetV3SmallBiFPNFCOS


@pytest.fixture
def model():
	m = MobileNetV3SmallBiFPNFCOS(num_classes=2, num_channels=128)
	m.arch_name = "test"
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
