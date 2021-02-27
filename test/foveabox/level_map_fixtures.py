import pytest

import os
import math

import torch

from test.fcos.level_map_fixtures import image_size


@pytest.fixture
def expected_joint_map_8x8(image_size):
	l = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 5 + [11., 43., 75.],
		[0] * 5 + [11., 43., 75.],
		[0] * 2 + [26., 58.] + [0] * 4,
		[0] * 2 + [26., 58.] + [0] * 4,
		[0] * 6 + [2., 34.],
		[0] * 6 + [2., 34.]
	]).unsqueeze(dim=-1)

	t = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 5 + [24., 24., 24.],
		[0] * 5 + [56., 56., 56.],
		[0] * 2 + [10., 10.] + [0] * 4,
		[0] * 2 + [42., 42.] + [0] * 4,
		[0] * 6 + [5., 5.],
		[0] * 6 + [37., 37.]
	]).unsqueeze(dim=-1)

	r = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 5 + [67., 35., 3.],
		[0] * 5 + [67., 35., 3.],
		[0] * 2 + [55., 23.] + [0] * 4,
		[0] * 2 + [55., 23.] + [0] * 4,
		[0] * 6 + [38., 6.],
		[0] * 6 + [38., 6.]
	]).unsqueeze(dim=-1)

	b = torch.tensor([
		[0] * 8,
		[0] * 8,
		[0] * 5 + [57., 57., 57.],
		[0] * 5 + [25., 25., 25.],
		[0] * 2 + [52., 52.] + [0] * 4,
		[0] * 2 + [20., 20.] + [0] * 4,
		[0] * 6 + [40., 40.],
		[0] * 6 + [8., 8.]
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
		[0.] * 5 + [1., 1., 1.],
		[0.] * 5 + [1., 1., 1.],
		[0.] * 8,
		[0.] * 8,
		[0.] * 8,
		[0.] * 8
	]).unsqueeze(dim=-1)

	fg2 = torch.tensor([
		[0.] * 8,
		[0.] * 8,
		[0.] * 8,
		[0.] * 8,
		[0.] * 2 + [1., 1.] + [0.] * 4,
		[0.] * 2 + [1., 1.] + [0.] * 4,
		[0.] * 6 + [1., 1.],
		[0.] * 6 + [1., 1.]
	]).unsqueeze(dim=-1)

	result = torch.cat([reg, bg, fg1, fg2], dim=-1)
	result = result.permute(2, 0, 1)

	return result


@pytest.fixture
def sample_reg_slab(image_size):
	l = torch.tensor([
		[-10] * 8,
		[-10] + [27., 59., 91., 123., 155., 187.] + [-10],
		[-10] + [27., 59., 91., 123., 155., 187.] + [-10],
		[-10] + [27., 59., 91., 123., 155., 187.] + [-10],
		[-10] + [27., 59., 91., 123., 155., 187.] + [-10],
		[-10] + [27., 59., 91., 123., 155., 187.] + [-10],
		[-10] + [27., 59., 91., 123., 155., 187.] + [-10],
		[-10] * 8
	]).unsqueeze(dim=-1)

	t = torch.tensor([
		[-10] * 8,
		[-10] + [21.] * 6 + [-10],
		[-10] + [53.] * 6 + [-10],
		[-10] + [85.] * 6 + [-10],
		[-10] + [117.] * 6 + [-10],
		[-10] + [149.] * 6 + [-10],
		[-10] + [181.] * 6 + [-10],
		[-10] * 8
	]).unsqueeze(dim=-1)

	r = torch.tensor([
		[-10] * 8,
		[-10] + [168., 136., 104., 72., 40., 8.] + [-10],
		[-10] + [168., 136., 104., 72., 40., 8.] + [-10],
		[-10] + [168., 136., 104., 72., 40., 8.] + [-10],
		[-10] + [168., 136., 104., 72., 40., 8.] + [-10],
		[-10] + [168., 136., 104., 72., 40., 8.] + [-10],
		[-10] + [168., 136., 104., 72., 40., 8.] + [-10],
		[-10] * 8
	]).unsqueeze(dim=-1)

	b = torch.tensor([
		[-10] * 8,
		[-10] + [178.] * 6 + [-10],
		[-10] + [146.] * 6 + [-10],
		[-10] + [114.] * 6 + [-10],
		[-10] + [82.] * 6 + [-10],
		[-10] + [50.] * 6 + [-10],
		[-10] + [18.] * 6 + [-10],
		[-10] * 8
	]).unsqueeze(dim=-1)

	reg = torch.cat([l, t, r, b], dim=-1)
	reg /= image_size

	return reg


@pytest.fixture
def expected_atss_11_joint_map_8x8(sample_reg_slab):
	l = sample_reg_slab[..., 0]
	t = sample_reg_slab[..., 1]
	r = sample_reg_slab[..., 2]
	b = sample_reg_slab[..., 3]

	bg = torch.tensor([
		[1.] * 8,
		[1.] + [0] * 6 + [1],
		[1.] + [0] * 6 + [1],
		[1.] + [0] * 6 + [1],
		[1.] + [0] * 6 + [1],
		[1.] + [0] * 6 + [1],
		[1.] + [0] * 6 + [1],
		[1.] * 8
	])

	atss_diff_raw = torch.abs(l - r) + torch.abs(t - b)
	neutral_map = atss_diff_raw.new_full(atss_diff_raw.shape, 2.)
	atss_diff = torch.where(bg > 0, neutral_map, atss_diff_raw)

	atss_flatten = atss_diff.flatten()

	_, indices = atss_flatten.topk(11, largest=False)
	z = torch.zeros_like(atss_flatten)

	result = z.scatter(-1, indices, 2)
	result = result.reshape([8, 8])
	result -= 1
	result = result.unsqueeze(-1)

	return result
