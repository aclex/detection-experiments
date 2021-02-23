import pytest

import math

import torch

from detector.fcos.unmap import Unmapper

from test.fcos.level_map_fixtures import (
	image_size,
	strides,
	sample,
	targets,
	expected_joint_map_8x8
)


@pytest.fixture
def unmapper(strides, image_size):
	m = Unmapper(
		strides, image_size, batch_size=1, num_classes=3)

	return m


@pytest.fixture
def expected_class():
	return 1


@pytest.fixture
def expected_labels(expected_class):
	label_list = [expected_class] * 8
	return torch.tensor(label_list)


@pytest.fixture
def expected_box1(image_size):
	return torch.tensor(
		[5., 11., 200., 210.], dtype=torch.float32)


@pytest.fixture
def expected_box2(image_size):
	return torch.tensor(
		[149., 40., 227., 121.], dtype=torch.float32)


@pytest.fixture
def expected_reg(expected_box1, expected_box2):
	return torch.stack([
		expected_box2,
		expected_box2,
		expected_box1,
		expected_box1,
		expected_box2,
		expected_box2,
		expected_box1,
		expected_box1
	])


def prefilter_mask(m, threshold):
	mx, _ = m[..., 1:].max(dim=-1)
	return mx >= threshold


def test_unmap_level(
		unmapper, expected_joint_map_8x8,
		expected_labels, expected_reg, image_size):
	pred_targets = unmapper._unmap_level(2, expected_joint_map_8x8.unsqueeze(
		dim=0))

	reg, cls = pred_targets

	mask = prefilter_mask(cls, 0.6)

	reg = reg[mask]
	cls = cls[mask]

	cls = cls.squeeze(dim=0)
	reg = reg.squeeze(dim=0)

	reg *= image_size

	labels = torch.argmax(cls, dim=-1)

	assert len(cls) == len(reg)
	assert len(reg) == len(expected_reg)

	assert torch.equal(labels, expected_labels)
	assert torch.allclose(reg, expected_reg)
