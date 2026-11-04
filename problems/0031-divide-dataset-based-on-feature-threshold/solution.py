import numpy as np

def divide_on_feature(X, feature_i, threshold):
	idxs = np.where(X[:, feature_i] >= threshold)
	other_idxs = np.where(X[:, feature_i] < threshold)
	return [X[idxs], X[other_idxs]]