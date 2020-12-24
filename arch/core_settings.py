import os
import json

import numpy as np


class CoreSettings():
	DEFAULT_MEAN = 128
	DEFAULT_STD = np.array([127, 127, 127])

	def __init__(self, config):
		try:
			self.settings = json.loads(config)
		except json.JSONDecodeError:
			if type(config) == str:
				with open(config, 'r') as f:
					self.settings = json.load(f)

		self.name = os.path.splitext(os.path.basename(config))[0]

		self.image_size = self.settings["image_size"]
		self.image_mean = self.settings.get("image_mean", self.DEFAULT_MEAN)
		self.image_std = self.settings.get("image_std", self.DEFAULT_STD)

