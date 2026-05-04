import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	means = np.array([np.mean(vec) for vec in vectors])
	substract = vectors - means.reshape(-1, 1)
	result = np.dot(substract, substract.T)*(1/(len(vectors[0]) - 1))
	return result.tolist()