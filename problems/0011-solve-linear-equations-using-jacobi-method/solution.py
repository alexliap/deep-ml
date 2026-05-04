import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	D = np.linalg.inv(np.diag(np.diag(A)))
	L = np.tril(A, -1)
	U = np.triu(A, 1)
	
	x = np.zeros_like(b).reshape(-1, 1)
	for i in range(n):
		x = np.dot(D, (np.array(b).reshape(-1, 1) - np.dot(L + U, x)))

	return np.round(x, 4).reshape(-1,)