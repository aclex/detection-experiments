import sys
import argparse

import torch

from storage.util import load


def main():
	parser = argparse.ArgumentParser(
		description="Utility to export model to ONNX")

	parser.add_argument("--model-path", '-p', type=str, required=True,
	                    help="path to the trained model")

	parser.add_argument("--output", '-o', type=str, default=None,
	                    help="output path, default is <model_path>.onnx")

	args = parser.parse_args()

	arch, model, class_names = load(args.model_path, inference=True)
	model.eval()

	if args.output is None:
		output_path = args.model_path + ".onnx"
	else:
		output_path = args.output
		if not output_path.endswith(".onnx"):
			output_path += ".onnx"

	dummy_input = torch.randn(1, 3, arch.image_size, arch.image_size).to(
		dtype=torch.float32)
	model.to(dtype=torch.float32)

	torch.onnx.export(model, dummy_input, output_path,
	                  input_names=arch.input_names(),
	                  output_names=arch.output_names(),
	                  opset_version=11,
	                  do_constant_folding=True,
	                  keep_initializers_as_inputs=True)


if __name__ == "__main__":
	try:
		main()

	except KeyboardInterrupt:
		sys.exit(0)
