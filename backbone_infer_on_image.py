import sys
import os

from PIL import Image

import torch
from torchvision import transforms

from backbone.mobilenetv3 import MobileNetV3_Small
from backbone.util import process_state_dict, mobilenetv3_key_rename

from backbone.imagenet_1000_classes import IMAGENET_1000_CLASSES


def load_image(path, img_size=(224, 224)):
	with open(path, 'rb') as f:
		image = Image.open(f)

		t = transforms.Compose([
			transforms.Resize(img_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								std=[0.229, 0.224, 0.225])
		])
		return t(image).unsqueeze(dim=0)


def main():
	if len(sys.argv) < 2:
		print("Usage: infer_on_image.py <path to image>")
		sys.exit(-1)

	input = load_image(sys.argv[1])

	model = MobileNetV3_Small()

	pack_path = os.path.join("backbone", "outlet", "mobilenetv3",
	                         "mbv3_small.pth.tar")

	pack = torch.load(pack_path, map_location="cpu")
	state_dict = process_state_dict(pack["state_dict"], mobilenetv3_key_rename)
	model.load_state_dict(state_dict, strict=True)
	model.eval()

	with torch.no_grad():
		v = model.forward(input)

	results = torch.topk(v, dim=-1, k=5)
	results = (results.values.squeeze().tolist(),
	           results.indices.squeeze().tolist())

	print("Top-5 results:")
	for i, r in enumerate(zip(*results)):
		print("%d. \"%s\", score: %.03f" %
		      (i + 1, IMAGENET_1000_CLASSES[r[1]], r[0]))


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
