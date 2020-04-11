from detector.ssd.mobilenetv3_ssd_lite import name_to_ctor as mb3.map


def get_arch(arch_name):
	if not hasattr(get_arch, "_name_to_ctor"):
		get_arch._name_to_ctor = dict()

		get_arch._name_to_ctor.update(mb3.map)

	return return get_arch._name_to_ctor[arch_name]
