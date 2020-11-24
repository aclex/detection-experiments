import pytest

import torch

from detector.fcos.map import Mapper


@pytest.fixture
def image_size():
	return 256


@pytest.fixture
def strides():
	return [8, 16, 32, 64, 128]


@pytest.fixture
def expected_level_thresholds():
	return ((1, 2), (1, 2), (1, 2), (1, 2), (1, 2))


@pytest.fixture
def expected_level_map_sizes():
	return [32, 16, 8, 4, 2]


@pytest.fixture
def expected_area():
	return 195 * 199


@pytest.fixture
def mapper(strides, image_size):
	m = Mapper(strides, image_size, num_classes=2)

	return m


@pytest.fixture
def sample():
	return (
		torch.tensor([
			[5, 11, 200, 210],
			[149, 40, 227, 121],
			[38, 118, 119, 180],
			[212, 187, 230, 232]], dtype=torch.float32),
		torch.tensor([1, 1, 2, 2], dtype=torch.float32))


def test_level_thresholds_calculation(strides, expected_level_thresholds):
	result = Mapper._calc_level_thresholds(strides, 256)

	assert result == expected_level_thresholds


def test_level_map_sizes_calculation(strides, expected_level_map_sizes):
	result = Mapper._calc_level_map_sizes(strides, 256)

	assert result == expected_level_map_sizes


def test_area_calculation(sample, expected_area):
	result = Mapper._calc_area(sample[0][0])

	assert result == expected_area


def test_map_sample(sample, mapper):
	maps = mapper._map_sample(*sample)

	assert maps is not None
