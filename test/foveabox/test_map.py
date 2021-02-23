import pytest

import math

import torch

from detector.foveabox.map import Mapper

from test.fcos.level_map_fixtures import (
	image_size,
	strides,
	sample,
	expected_level_map_sizes
)

from test.foveabox.level_map_fixtures import (
	expected_joint_map_8x8
)


@pytest.fixture
def expected_level_thresholds(expected_level_map_sizes, image_size):
	pixel_sizes = [1 / float(e) for e in expected_level_map_sizes]
	level_thresholds = ((1, 4), (2, 4), (2, 4), (2, 4))

	return tuple(
		(l[0] * p, l[1] * p) for l, p in zip(level_thresholds, pixel_sizes))


@pytest.fixture
def mapper(strides, image_size):
	m = Mapper(strides, image_size, sigma=1.0, num_classes=3)

	return m


def test_level_thresholds_calculation(strides, expected_level_thresholds):
	result = Mapper._calc_level_thresholds(strides, 256)

	assert result == expected_level_thresholds


def test_map_sample(sample, mapper, expected_joint_map_8x8):
	maps = mapper._map_sample(*sample)

	assert maps is not None

	result_joint_map_8x8 = maps[2]

	assert result_joint_map_8x8.shape == expected_joint_map_8x8.shape

	assert torch.allclose(result_joint_map_8x8, expected_joint_map_8x8)