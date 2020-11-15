import pytest

import torch

from fpn.bifpn import SoftMaxWeightedSum, HardWeightedSum, BiFPN


@pytest.fixture
def input():
	return [
		torch.full((1, 1, 32, 32), 3.), # P3
		torch.full((1, 2, 16, 16), 4.), # P4
		torch.full((1, 3, 8, 8), 5.),   # P5
		torch.full((1, 4, 4, 4), 6.),   # P6
		torch.full((1, 5, 2, 2), 7.)    # P7
	]


@pytest.fixture
def w_input(input):
	row = torch.full((1, 3, 8, 8), 5.)   # P5
	return torch.stack([row, row, row])


@pytest.fixture
def hard_sum():
	return HardWeightedSum(3)


@pytest.fixture
def soft_sum():
	return SoftMaxWeightedSum(3)


@pytest.fixture
def model():
	return BiFPN(
		feature_channels=[1, 2, 3, 4, 5],
		feature_strides=[2, 4, 8, 16, 32],
		out_channels=3, num_layers=3)


def test_hard_sum(hard_sum, w_input):
	result = hard_sum.forward(w_input)

	assert result.shape == (1, 3, 8, 8)


def test_soft_sum(soft_sum, w_input):
	result = soft_sum.forward(w_input)

	assert result.shape == (1, 3, 8, 8)


def test_bifpn(model, input):
	result = model.forward(input)

	assert len(result) == len(input)


def test_bifpn(model, input):
	result = model.forward(input)

	assert len(result) == len(input)

	for i in range(len(result)):
		assert result[i].shape[0] == input[i].shape[0]
		assert result[i].shape[2:] == input[i].shape[2:]
