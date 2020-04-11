import torch

from storage.factory import arch


def load(model_file, device=None):
	pack = torch.load(model_file, map_location=device)

	ctor = arch(pack["arch"])
	class_map = pack["class_map"]

	model = ctor(len(class_map))

	model.load_state_dict(pack["state_dict"], strict=True)
	model.to(device)

	return model, class_map


def save(model, class_map, filename):
	pack = {
		"arch": model.arch_name,
		"class_map": class_map,
		"state_dict": model.state_dict()
	}

	torch.save(pack, filename)
