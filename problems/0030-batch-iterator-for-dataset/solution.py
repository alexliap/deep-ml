import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	# Your code here
	data_length = X.shape[0]
	num_batches = (X.shape[0] // batch_size) + (X.shape[0] % batch_size)
	for i in range(num_batches):
		start = i*batch_size
		end = start + batch_size
		x_batch = X[start:end, :]
		if y is not None:
			y_batch = y.reshape(1, -1)[:, start:end]
			yield [x_batch.tolist(), y_batch.tolist()[0]]
		else:
			yield [x_batch.tolist()]