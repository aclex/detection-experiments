import pytest

from arch.compound_scaling import CompoundScaling


@pytest.fixture
def uneven_image_size():
	return 424


@pytest.fixture
def input_pheta():
	return 1


@pytest.fixture
def expected_pheta():
	return 0


@pytest.fixture
def expected_height():
	return 4


def test_pheta(uneven_image_size, expected_pheta):
	pheta = CompoundScaling.pheta(uneven_image_size)

	assert pheta == expected_pheta


def test_fpn_width(input_pheta):
	w = CompoundScaling.fpn_width(input_pheta)

	assert w % 8 == 0


def test_fpn_height(input_pheta, expected_height):
	h = CompoundScaling.fpn_height(input_pheta)

	assert h == expected_height
