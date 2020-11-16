import pytest

import torch

from detector.fcos.model import MobileNetV3SmallBiFPNFCOS


@pytest.fixture
def model():
	return MobileNetV3SmallBiFPNFCOS(num_classes=2)


def test_sanity(model):
	a = torch.ones((2, 3, 512, 512))

	r = model.forward(a)

	assert len(r) == model.head.num_levels
