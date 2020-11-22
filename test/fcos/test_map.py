import pytest

import torch

from detector.fcos.map import Mapper


@pytest.fixture
def strides():
	return [8, 16, 32, 64, 128]


@pytest.fixture
def expected_level_thresholds():
	return (
		((1, 2), (1, 2), (1, 2), (1, 2), (1, 2)),
		((1, 2), (1, 2), (1, 2), (1, 2), (1, 2)))


def test_level_thresholds_calculation(strides, expected_level_thresholds):
	result = Mapper._calc_level_thresholds(strides, (256, 256))

	assert result == expected_level_thresholds
