def hard_sigmoid(x: float) -> float:
	"""
	Implements the Hard Sigmoid activation function.

	Args:
		x (float): Input value

	Returns:
		float: The Hard Sigmoid of the input
	"""
	# Your code here
	return float(max(0, min(1, 0.2*x + 0.5)))