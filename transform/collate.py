import torch


def collate(batch):
	images = []
	bboxes = []
	labels = []

	for pack in batch:
		images.append(pack["image"])
		bboxes.append(pack["bboxes"])
		labels.append(pack["category_id"])

	return {
		"image": torch.stack(images),
		"bboxes": bboxes,
		"category_id": labels
	}
