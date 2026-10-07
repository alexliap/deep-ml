import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	score = np.matmul(Q, K.T)/np.sqrt(K.shape[1]) + mask
	soft_nom = np.exp(score - np.max(score, axis=-1, keepdims=True))
	softmax = soft_nom / np.sum(soft_nom, axis=-1, keepdims=True)

	return np.matmul(softmax, V)