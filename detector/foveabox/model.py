import functools

from detector.foveabox.head import Head
from detector.foveabox.unmap import Unmapper

from detector.fcos import model


Blueprint = functools.partial(model.Blueprint, head=Head)
BlueprintInference = functools.partial(
	model.BlueprintInference, head=Head, unmapper=Unmapper)
