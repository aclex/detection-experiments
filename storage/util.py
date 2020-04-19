import torch

from storage.factory import get_arch


def load(model_file, batch_size=None, inference=False, device=None):
	pack = torch.load(model_file, map_location=device)

	ctor = get_arch(pack["arch"])
	class_names = pack["class_names"]

	model = ctor(len(class_names), batch_size=batch_size, inference=inference)

	model.load_state_dict(pack["state_dict"], strict=True)
	model.to(device)

	return model, class_names


def save(model, class_names, filename):
	pack = {
		"arch": model.arch_name,
		"class_names": class_names,
		"state_dict": model.state_dict()
	}

	torch.save(pack, filename)
