import numpy as np


class CoreSettings():
	DEFAULT_MEAN = 128
	DEFAULT_STD = np.array([127, 127, 127])

	def __init__(self):
		self.image_size = self.settings["image_size"]
		self.image_mean = self.settings.get("image_mean", self.DEFAULT_MEAN)
		self.image_std = self.settings.get("image_std", self.DEFAULT_STD)

