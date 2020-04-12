import sys
import os

from tqdm import tqdm


def interactive(iterable):
	# check if we're not under GNU screen and have interactive shell
	if sys.stdout.isatty() and "STY" not in os.environ:
		return tqdm(iterable)
	else:
		return iterable
