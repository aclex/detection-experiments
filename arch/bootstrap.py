import os

import importlib

import json


def _igetattr(obj, attr):
    for a in dir(obj):
        if a.lower() == attr.lower():
            return getattr(obj, a)


def get_arch_name(config):
	with open(config, 'r') as f:
		settings = json.load(f)

		return settings["arch"]


def get_arch(config):
	name = get_arch_name(config)
	spec = importlib.util.spec_from_file_location(
		name,
		os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py"))

	arch = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(arch)

	arch_class = _igetattr(arch, name)

	return arch_class(config)
