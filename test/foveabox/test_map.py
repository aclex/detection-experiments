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
	expected_joint_map_8x8,
	sample_reg_slab,
	expected_atss_11_joint_map_8x8,
	sample_atss_box
)


@pytest.fixture
def expected_level_thresholds(expected_level_map_sizes, image_size):
	pixel_sizes = [1 / float(e) for e in expected_level_map_sizes]
	level_thresholds = ((1, 4), (1, 4), (1, 4), (1, 4))

	return tuple(
		(l[0] * p, l[1] * p) for l, p in zip(level_thresholds, pixel_sizes))


@pytest.fixture
def mapper(strides, image_size):
	m = Mapper(strides, image_size, sigma=1.0, num_classes=3)

	return m


@pytest.fixture
def atss_mapper(strides, image_size):
	m = Mapper(strides, image_size, atss_k=11, num_classes=3)

	return m


@pytest.fixture
def fake_atss_mapper(strides, image_size):
	m = Mapper(strides, image_size, atss_k=image_size ** 2, num_classes=3)

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


def test_calc_atss(sample_reg_slab, atss_mapper, expected_atss_11_joint_map_8x8):
	atss_slab = atss_mapper._calc_atss_slab(sample_reg_slab)

	assert atss_slab.shape == expected_atss_11_joint_map_8x8.shape

	assert torch.allclose(atss_slab, expected_atss_11_joint_map_8x8)


def test_atss_sanity(sample_reg_slab, sample_atss_box, mapper, fake_atss_mapper):
	atss_slab = fake_atss_mapper._calc_atss_slab(sample_reg_slab)
	fovea_slab = mapper._calc_fovea_slab(sample_atss_box, sample_reg_slab)

	rectified_atss_slab = fake_atss_mapper._filter_background(atss_slab)
	rectified_fovea_slab = mapper._filter_background(fovea_slab)

	assert torch.equal(rectified_atss_slab, rectified_fovea_slab)
