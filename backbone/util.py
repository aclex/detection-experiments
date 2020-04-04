def process_state_dict(state_dict, func):
	return dict({ func(k, v) for k, v in state_dict.items() })


def mobilenetv3_key_rename(k, v):
	return k.replace("module.", ""), v
