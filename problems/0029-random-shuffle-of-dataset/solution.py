import numpy as np

def shuffle_data(X, y, seed=None):
	np.random.seed(seed)
	dataset_len = len(y)
	indices = np.arange(dataset_len)
	np.random.shuffle(indices)

	return X[indices], y[indices]