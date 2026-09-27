import numpy as np

def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	nominator = np.sum(np.array(v)*np.array(L))
	denominator = np.sum(np.array(L)*np.array(L))
	return (nominator/denominator)*np.array(L)
