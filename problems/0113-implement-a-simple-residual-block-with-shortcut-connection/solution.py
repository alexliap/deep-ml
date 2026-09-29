import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	x_1 = np.maximum(np.matmul(w1, x), 0)
	x_2 = np.matmul(w2, x_1) + x
	return np.maximum(x_2, 0)