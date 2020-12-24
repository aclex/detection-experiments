import os
import json

import numpy as np

from predict.predictor import Predictor


def json_from_file_or_string(config):
	try:
		return json.loads(config)
	except json.JSONDecodeError:
		with open(config, 'r') as f:
			return json.load(f)


class CoreSettings():
	DEFAULT_MEAN = 128
	DEFAULT_STD = np.array([127, 127, 127])

	def __init__(self, config):
		self.settings = json_from_file_or_string(config)

		self.name = os.path.splitext(os.path.basename(config))[0]

		self.image_size = self.settings["image_size"]
		self.image_mean = self.settings.get("image_mean", self.DEFAULT_MEAN)
		self.image_std = self.settings.get("image_std", self.DEFAULT_STD)

	def predictor(self, net, device=None):
		return Predictor(arch=self, net=net, device=device)
