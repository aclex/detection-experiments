import sys
import argparse

import cv2

from detector.ssd.utils.misc import Timer
from detector.ssd.mobilenetv3_ssd_lite import (
	create_mobilenetv3_large_ssd_lite,
	create_mobilenetv3_small_ssd_lite,
	create_mobilenetv3_ssd_lite_predictor
)

from storage.util import load


def main():
	parser = argparse.ArgumentParser("Utility to process an image "
									 "through the detection model")

	parser.add_argument("--model-path", '-p', type=str, required=True,
						help="path to the trained model")
	parser.add_argument("video_path", type=str, nargs='?',
						help="video to process (use camera if omitted)")

	args = parser.parse_args()

	if len(args.video_path) > 0:
		cap = cv2.VideoCapture(args.video_path)  # capture from file
	else:
		cap = cv2.VideoCapture(0)   # capture from camera


	model, class_names = load(args.model_path)
	model.eval()

	predictor = create_mobilenetv3_ssd_lite_predictor(model, candidate_size=200)

	timer = Timer()

	while True:
		ret, orig_image = cap.read()

		if not ret or orig_image is None:
			break

		image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

		timer.start()

		boxes, labels, probs = predictor.predict(image, 10, 0.4)

		interval = timer.end()
		print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

		for i in range(boxes.size(0)):
			box = boxes[i, :]
			cv2.rectangle(orig_image,
						  (box[0], box[1]), (box[2], box[3]),
						  (255, 255, 0), 4)

			label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
			cv2.putText(orig_image, label,
						(box[0] + 20, box[1] + 40),
						cv2.FONT_HERSHEY_SIMPLEX,
						1,  # font scale
						(255, 0, 255),
						2)  # line type

		cv2.imshow('annotated', orig_image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
