import os

import importlib

import json

from arch.core_settings import json_from_file_or_string


def _igetattr(obj, attr):
    for a in dir(obj):
        if a.lower() == attr.lower():
            return getattr(obj, a)


def get_arch_name(config):
	settings = json_from_file_or_string(config)

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
