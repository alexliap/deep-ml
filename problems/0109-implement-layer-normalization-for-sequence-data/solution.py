import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	mean_x = np.mean(X, axis=-1, keepdims=True)
	var_x = np.var(X, axis=-1, keepdims=True)

	return beta + gamma * (X-mean_x) / np.sqrt(var_x + epsilon)