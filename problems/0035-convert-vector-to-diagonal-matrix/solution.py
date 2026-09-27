import numpy as np

def make_diagonal(x):
	# Your code here
	diag = np.zeros((x.shape[0], x.shape[0]))
	for i in range(len(x)):
		diag[i, i] = x[i]

	return diag