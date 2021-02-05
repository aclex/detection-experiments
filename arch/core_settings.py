import os
import json

import numpy as np


def json_from_file_or_string(config):
	try:
		return json.loads(config)
	except json.JSONDecodeError:
		with open(config, 'r') as f:
			return json.load(f)


class CoreSettings():
	def __init__(self, config):
		self.settings = json_from_file_or_string(config)

		self.name = os.path.splitext(os.path.basename(config))[0]

		self.image_size = self.settings["image_size"]

	@staticmethod
	def input_names():
		return ["img"]

	@staticmethod
	def output_names():
		return ["cls", "reg"]
