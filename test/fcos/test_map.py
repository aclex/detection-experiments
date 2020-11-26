import pytest

import math

import torch

from detector.fcos.map import Mapper


@pytest.fixture
def image_size():
	return 256


@pytest.fixture
def strides():
	return [8, 16, 32, 64]


@pytest.fixture
def expected_level_thresholds():
	return ((2, 4), (2, 4), (2, 4), (2, 4))


@pytest.fixture
def expected_level_map_sizes():
	return [32, 16, 8, 4]


@pytest.fixture
def expected_area():
	return 195 * 199


@pytest.fixture
def mapper(strides, image_size):
	m = Mapper(strides, image_size, num_classes=3)

	return m


@pytest.fixture
def sample():
	return (
		torch.tensor([
			[5, 11, 200, 210],
			[149, 40, 227, 121],
			[38, 118, 119, 180],
			[190, 187, 230, 232]], dtype=torch.float32),
		torch.tensor([1, 1, 2, 2], dtype=torch.float32))


@pytest.fixture
def expected_joint_map_8x8():
	l = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 8,
		[0] * 3 + [91., 123.] + [0] * 3,
		[0] * 3 + [91., 123.] + [0] * 3,
		[0] * 8,
		[0] * 8,
		[0] * 8
	]).unsqueeze(dim=-1)

	t = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 8,
		[0] * 3 + [85., 85.] + [0] * 3,
		[0] * 3 + [117., 117.] + [0] * 3,
		[0] * 8,
		[0] * 8,
		[0] * 8
	]).unsqueeze(dim=-1)

	r = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 8,
		[0] * 3 + [104., 72.] + [0] * 3,
		[0] * 3 + [104., 72.] + [0] * 3,
		[0] * 8,
		[0] * 8,
		[0] * 8
	]).unsqueeze(dim=-1)

	b = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 8,
		[0] * 3 + [114., 114.] + [0] * 3,
		[0] * 3 + [82., 82.] + [0] * 3,
		[0] * 8,
		[0] * 8,
		[0] * 8
	]).unsqueeze(dim=-1)

	reg = torch.cat([l, t, r, b], dim=-1)
	reg /= 32

	bg = torch.zeros([8, 8, 1])
	fg1 = torch.tensor([
		[0.] * 8,
		[0.] * 8,
		[0.] * 8,
		[0.] * 3 + [1., 1.] + [0.] * 3,
		[0.] * 3 + [1., 1.] + [0.] * 3,
		[0.] * 8,
		[0.] * 8,
		[0.] * 8
	]).unsqueeze(dim=-1)

	fg2 = torch.zeros([8, 8, 1])

	centerness = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 8,
		[0] * 3 +
			[
				math.sqrt(91. / 104. * 85. / 114.),
				math.sqrt(72. / 123. * 85. / 114.)] +
			[0] * 3,
		[0] * 3 +
			[
				math.sqrt(91. / 104. * 82. / 117.),
				math.sqrt(72. / 123. * 82. / 117.)] +
			[0] * 3,
		[0] * 8,
		[0] * 8,
		[0] * 8
	]).unsqueeze(dim=-1)

	result = torch.cat([reg, centerness, bg, fg1, fg2], dim=-1)

	return result



def test_level_thresholds_calculation(strides, expected_level_thresholds):
	result = Mapper._calc_level_thresholds(strides, 256)

	assert result == expected_level_thresholds


def test_level_map_sizes_calculation(strides, expected_level_map_sizes):
	result = Mapper._calc_level_map_sizes(strides, 256)

	assert result == expected_level_map_sizes


def test_area_calculation(sample, expected_area):
	result = Mapper._calc_area(sample[0][0])

	assert result == expected_area


def test_map_sample(sample, mapper, expected_joint_map_8x8):
	maps = mapper._map_sample(*sample)

	assert maps is not None

	result_joint_map_8x8 = maps[2]

	assert result_joint_map_8x8.shape == expected_joint_map_8x8.shape

	assert torch.allclose(result_joint_map_8x8, expected_joint_map_8x8)
