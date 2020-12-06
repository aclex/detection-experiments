import pytest

import torch

from loss.ciou import box_ciou


@pytest.fixture
def x():
	return torch.tensor([[1, 1, 4, 5]], dtype=torch.float32)


@pytest.fixture
def y_separated():
	return torch.tensor([[10, 12, 17, 20]], dtype=torch.float32)


@pytest.fixture
def y_overlapped():
	return torch.tensor([[1, 1, 5, 5]], dtype=torch.float32)


@pytest.fixture
def expected_separated():
	return torch.tensor([[1.470016207]])


@pytest.fixture
def expected_overlapped():
	return torch.tensor([[0.258070443]])


def test_ciou_separated(x, y_separated, expected_separated):
	r = box_ciou(x, y_separated)

	assert torch.allclose(r, expected_separated)

def test_ciou_overlapped(x, y_overlapped, expected_overlapped):
	r = box_ciou(x, y_overlapped)

	assert torch.allclose(r, expected_overlapped)
