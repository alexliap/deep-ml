def calculate_parameters(layers: list[dict]) -> int:
	"""
	Calculate the total number of trainable parameters in a neural network.

	Args:
		layers: List of dictionaries, each describing a layer.

	Returns:
		Total number of trainable parameters (int).
	"""
	params = 0
	for layer in layers:
		if layer["type"] == "dense":
			params += layer["input_size"] * layer["output_size"]
			if layer.get("bias", True):
				params += layer["output_size"]
		else: # conv
			params += layer["in_channels"] * layer["out_channels"] * layer["kernel_size"]**2
			if layer.get("bias", True):
				params += layer["out_channels"]

	return params
