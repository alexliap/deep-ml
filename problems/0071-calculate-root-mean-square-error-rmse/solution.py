import numpy as np

def rmse(y_true, y_pred):
	# Write your code here
	rmse = np.sqrt(np.mean((y_pred - y_true)**2))
	return round(rmse, 3)
