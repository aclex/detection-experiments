import time


class Timer:
	def __init__(self):
		self.clock = {}

	def start(self, key="default"):
		self.clock[key] = time.time()

	def end(self, key="default"):
		if key not in self.clock:
			raise Exception(f"{key} is not in the clock.")
		interval = time.time() - self.clock[key]
		del self.clock[key]
		return interval


def store_labels(path, labels):
	with open(path, "w") as f:
		f.write("\n".join(labels))
