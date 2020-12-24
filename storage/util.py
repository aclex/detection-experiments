import json

import torch

from storage.factory import get_arch


def load(model_file, batch_size=None, inference=False, device=None):
	pack = torch.load(model_file, map_location=device)

	arch = get_arch(pack["config"])
	class_names = pack["class_names"]

	model = arch.build(
		len(class_names), batch_size=batch_size, inference=inference)

	model.load_state_dict(pack["state_dict"], strict=True)
	model.to(device)

	return model, class_names


def save(arch, model, class_names, filename):
	config = json.dumps(arch.settings)

	pack = {
		"config": config,
		"class_names": class_names,
		"state_dict": model.state_dict()
	}

	torch.save(pack, filename)
