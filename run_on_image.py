import sys
import argparse

import cv2

from detector.ssd.utils.misc import Timer
from detector.ssd.mobilenetv3_ssd_lite import (
	create_mobilenetv3_large_ssd_lite,
	create_mobilenetv3_small_ssd_lite,
	create_mobilenetv3_ssd_lite_predictor
)


def main():
	parser = argparse.ArgumentParser("Utility to process an image "
									 "through the detection model")

	parser.add_argument("--model", '-m', type=str, default='mb3-small-ssd-lite',
						help="model to use ('mb3-large-ssd-lite', "
						"'mb3-small-ssd-lite' are supported)")
	parser.add_argument("--model-path", '-p', type=str, required=True,
						help="path to the trained model")
	parser.add_argument("image_path", type=str, nargs=1,
						help="image to process")

	args = parser.parse_args()

	class_names = ('BACKGROUND',
			'aeroplane', 'bicycle', 'bird', 'boat',
			'bottle', 'bus', 'car', 'cat', 'chair',
			'cow', 'diningtable', 'dog', 'horse',
			'motorbike', 'person', 'pottedplant',
			'sheep', 'sofa', 'train', 'tvmonitor')

	if args.model == 'mb3-large-ssd-lite':
		net = create_mobilenetv3_large_ssd_lite(len(class_names), is_test=True)
	elif args.model == 'mb3-small-ssd-lite':
		net = create_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
	else:
		print("Model type is wrong. It should be one of mb3-large-ssd-lite "
			  "or mb3-small-ssd-lite.")
		sys.exit(1)

	net.load(args.model_path)

	predictor = create_mobilenetv3_ssd_lite_predictor(net, candidate_size=200)

	orig_image = cv2.imread("airplane.jpg")
	image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
	boxes, labels, probs = predictor.predict(image, 10, 0.4)

	for i in range(boxes.size(0)):
		box = boxes[i, :]
		cv2.rectangle(orig_image,
					  (box[0], box[1]), (box[2], box[3]),
					  (255, 255, 0), 4)
		label = f"{labels[i]}: {probs[i]:.2f}"
		cv2.putText(orig_image, label,
					(box[0] + 20, box[1] + 40),
					cv2.FONT_HERSHEY_SIMPLEX,
					1,  # font scale
					(255, 0, 255),
					2)  # line type

	path = "run_ssd_example_output.jpg"
	cv2.imwrite(path, orig_image)
	cv2.imshow("result", orig_image)
	print(f"Found {len(probs)} objects. The output image is {path}")
	cv2.waitKey(0)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
