import pytest

import torch

from .level_map_fixtures import strides, image_size, output_sample

from detector.fcos.loss import Loss


@pytest.fixture
def loss(strides, image_size):
	return Loss(strides=strides, image_size=image_size, num_classes=3)


@pytest.fixture
def expected():
	return torch.tensor([0.8143])


def test_sanity(loss, output_sample, expected):
	r = loss.forward(output_sample, output_sample)

	assert torch.allclose(r, expected, rtol=1e-4)
