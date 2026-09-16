def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	import numpy as np

	P = np.matmul(np.linalg.inv(np.array(C)), np.array(B))
	return P