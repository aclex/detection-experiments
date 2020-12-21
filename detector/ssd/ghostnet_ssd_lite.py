import torch

from backbone.ghostnet import GhostNet

from detector.ssd.ssd import SSD, SSDInference
import detector.ssd.config as config

from predict.predictor import Predictor


def create_ghostnet_ssd_lite(num_classes,
                             pretrained=False,
                             batch_size=None,
                             inference=False):
	base_net = GhostNet(pretrained=pretrained)

	if not inference:
		return SSD(num_classes, base_net, "ghostnet-ssd-lite",
				   batch_size=batch_size, config=config)
	else:
		return SSDInference(num_classes, base_net, "ghostnet-ssd-lite",
							batch_size=batch_size, config=config)


name_to_ctor = {
	"ghostnet-ssd-lite": create_ghostnet_ssd_lite
}
