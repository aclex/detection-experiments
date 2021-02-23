import pytest

import os
import math

import torch


@pytest.fixture
def image_size():
	return 256


@pytest.fixture
def strides():
	return [8, 16, 32, 64]


@pytest.fixture
def sample(image_size):
	return (
		torch.tensor([
			[5, 11, 200, 210],
			[149, 40, 227, 121],
			[38, 118, 119, 180],
			[190, 187, 230, 232]], dtype=torch.float32) / image_size,
		torch.tensor([1, 1, 2, 2], dtype=torch.float32))


@pytest.fixture
def output_sample():
	d = os.path.dirname(os.path.abspath(__file__))
	filename = "map_output_sample.pt"

	r = torch.load(os.path.join(d, filename))

	for c in r:
		c.requires_grad_(False)

	return r


@pytest.fixture
def targets(sample):
	return ([sample[0]], [sample[1]])


@pytest.fixture
def expected_level_map_sizes():
	return [32, 16, 8, 4]


@pytest.fixture
def expected_joint_map_8x8(image_size):
	l = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 5 + [11., 0, 75.],
		[0] * 3 + [91., 123., 11., 0, 75.],
		[0] * 3 + [91., 123.] + [0] * 3,
		[0] * 8,
		[0] * 8,
		[0] * 8
	]).unsqueeze(dim=-1)

	t = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 5 + [24., 0, 24.],
		[0] * 3 + [85., 85., 56, 0, 56.],
		[0] * 3 + [117., 117.] + [0] * 3,
		[0] * 8,
		[0] * 8,
		[0] * 8
	]).unsqueeze(dim=-1)

	r = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 5 + [67., 0, 3.],
		[0] * 3 + [104., 72., 67., 0., 3.],
		[0] * 3 + [104., 72.] + [0] * 3,
		[0] * 8,
		[0] * 8,
		[0] * 8
	]).unsqueeze(dim=-1)

	b = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 5 + [57., 0, 57.],
		[0] * 3 + [114., 114., 25., 0, 25.],
		[0] * 3 + [82., 82.] + [0] * 3,
		[0] * 8,
		[0] * 8,
		[0] * 8
	]).unsqueeze(dim=-1)

	reg = torch.cat([l, t, r, b], dim=-1)
	reg /= image_size

	bg = torch.tensor([
		[1.] * 8,
		[1.] + [0.] * 6 + [1.],
		[1.] + [0.] * 7,
		[1.] + [0.] * 7,
		[1.] + [0.] * 6 + [1.],
		[1.] + [0.] * 6 + [1.],
		[1.] + [0.] * 7,
		[1.] * 6 + [0.] * 2
	]).unsqueeze(dim=-1)

	fg1 = torch.tensor([
		[0.] * 8,
		[0.] * 8,
		[0.] * 5 + [1., 0., 1.],
		[0.] * 3 + [1., 1.] + [1., 0., 1.],
		[0.] * 3 + [1., 1.] + [0.] * 3,
		[0.] * 8,
		[0.] * 8,
		[0.] * 8
	]).unsqueeze(dim=-1)

	fg2 = torch.zeros([8, 8, 1])

	centerness = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 5 +
			[
				math.sqrt(11. / 67. * 24. / 57.), 0.,
				math.sqrt(3. / 75. * 24. / 57.)],
		[0] * 3 +
			[
				math.sqrt(91. / 104. * 85. / 114.),
				math.sqrt(72. / 123. * 85. / 114.),
				math.sqrt(11. / 67. * 25. / 56.), 0.,
				math.sqrt(3. / 75. * 25. / 56.)],
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
	result = result.permute(2, 0, 1)

	return result
