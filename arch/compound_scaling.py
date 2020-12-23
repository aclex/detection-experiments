import math


class CompoundScaling:
	PHETA_FOR_IMAGE_SIZE = {
		256: -1,
		512: 0,
		640: 1,
		768: 2,
		896: 3,
		1024: 4,
		1280: 5,
		1408: 6,
		1536: 7
	}

	@staticmethod
	def fpn_width(pheta):
		w = 64 * 1.35 ** pheta

		return int(math.ceil(w / 8.) * 8)

	@staticmethod
	def fpn_depth(pheta):
		d = 2 + pheta

		return d

	@staticmethod
	def fpn_height(pheta):
		if pheta < 0:
			return 3

		elif pheta < 3:
			return 4

		else:
			return 5

	@staticmethod
	def head_depth(pheta):
		if pheta < 0:
			return 2

		elif pheta < 3:
			return 3

		elif pheta < 6:
			return 4

		else:
			return 5

	@staticmethod
	def pheta(image_size):
		k, v = min(
			CompoundScaling.PHETA_FOR_IMAGE_SIZE.items(),
			key=lambda i: abs(i[0] - image_size))

		return v


