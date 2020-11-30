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
		strides, image_size, batch_size=1, num_classes=3,
		prefilter_threshold=0.1)

	return m


@pytest.fixture
def expected_class():
	return 1


@pytest.fixture
def expected_labels(expected_class):
	label_list = [expected_class] * 4
	return torch.tensor(label_list)


@pytest.fixture
def expected_box():
	return torch.tensor([5., 11., 200., 210.], dtype=torch.float32)


@pytest.fixture
def expected_reg(expected_box):
	return expected_box.unsqueeze(dim=0).expand(4, -1)


def test_unmap_level(
		unmapper, expected_joint_map_8x8,
		expected_labels, expected_reg):
	pred_targets = unmapper._unmap_level(2, expected_joint_map_8x8)

	reg, cls = pred_targets
	labels = torch.argmax(cls, dim=-1)

	assert len(cls) == len(reg)
	assert len(reg) == 4

	assert torch.equal(labels, expected_labels)
	assert torch.allclose(reg, expected_reg)
