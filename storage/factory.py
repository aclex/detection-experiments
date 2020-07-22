from detector.ssd import mobilenetv3_ssd_lite, ghostnet_ssd_lite


def get_arch(arch_name):
	if not hasattr(get_arch, "_name_to_ctor"):
		get_arch._name_to_ctor = dict()

		get_arch._name_to_ctor.update(mobilenetv3_ssd_lite.name_to_ctor)
		get_arch._name_to_ctor.update(ghostnet_ssd_lite.name_to_ctor)

	return get_arch._name_to_ctor[arch_name]
