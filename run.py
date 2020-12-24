import sys
import argparse

import cv2

import torch

from detector.ssd.utils.misc import Timer

from arch.bootstrap import get_arch

from storage.util import load


def draw_predictions(frame, boxes, labels, scores, class_names):
	for i in range(boxes.size(0)):
		box = boxes[i, :]
		cv2.rectangle(frame,
		              (box[0], box[1]), (box[2], box[3]),
		              (255, 255, 0), 4)

		label = f"{class_names[labels[i]]}: {scores[i]:.2f}"
		cv2.putText(frame, label,
		            (box[0] + 20, box[1] + 40),
		            cv2.FONT_HERSHEY_SIMPLEX,
		            1,  # font scale
		            (255, 0, 255),
		            2)  # line type


def predict_and_show(orig_image, predictor, class_names, timer):
	image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

	timer.start("inference")
	boxes, labels, probs = predictor.predict(image, 10, 0.4)
	interval = timer.end("inference")
	print(f'Inference time: {interval:.3f}s, '
	      f'Detect Objects: {labels.size(0)}.')

	draw_predictions(orig_image, boxes, labels, probs, class_names)

	cv2.imshow("result", orig_image)


def main():
	parser = argparse.ArgumentParser("Utility to process an image "
	                                 "through the detection model")

	parser.add_argument("--model-path", '-p', type=str, required=True,
	                    help="path to the trained model")
	parser.add_argument("--image", '-i', action='store_true',
	                    help="process on image")
	parser.add_argument("--video", '-v', action='store_true',
	                    help="process on video")
	parser.add_argument('--device', type=str, help='device to use')
	parser.add_argument('--output', '-o', type=str,
	                    help="save the results to the specified file")
	parser.add_argument("path", type=str, nargs='?',
	                    help="file to process (use camera if omitted and "
	                    "'--video' is set")

	args = parser.parse_args()

	if args.device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		device = args.device

	if device.startswith("cuda"):
		logging.info("Use CUDA")

	if args.image and args.video:
		print("Can process either image or video, but not both")
		sys.exit(-1)

	model, class_names = load(args.model_path, device=device,
	                          inference=True)
	model.eval()

	predictor = create_mobilenetv3_ssd_lite_predictor(
		model, candidate_size=200,
		device=device)

	timer = Timer()

	if args.image:
		orig_image = cv2.imread(args.path)
		predict_and_show(orig_image, predictor, class_names, timer)

		if args.output is not None:
			cv2.imwrite(orig_image, args.output)

		cv2.waitKey(0)

	elif args.video:
		if len(args.path) > 0:
			cap = cv2.VideoCapture(args.path)  # capture from file
		else:
			cap = cv2.VideoCapture(0)   # capture from camera

		out = None

		if args.output is not None:
			frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps = int(cap.get(cv2.CAP_PROP_FPS))
			out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
			                      fps, (frame_width, frame_height))

		while True:
			ret, orig_image = cap.read()

			if not ret or orig_image is None:
				break

			predict_and_show(orig_image, predictor, class_names, timer)

			if out is not None:
				print("writing to")
				out.write(orig_image)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cap.release()

		if out is not None:
			out.release()

	cv2.destroyAllWindows()


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(0)
