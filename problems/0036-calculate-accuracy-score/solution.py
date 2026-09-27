import numpy as np

def accuracy_score(y_true, y_pred):
	acc = []
	for i in range(len(y_true)):
		hit = y_true[i] == y_pred[i]
		acc.append(hit)
	return np.mean(acc)