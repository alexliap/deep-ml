import numpy as np

def dice_score(y_true, y_pred):
	inter = y_true + y_pred
	if np.sum(inter) == 0:
		return 0.0
	tp = np.sum(inter == 2)
	fp_fn = np.sum(inter == 1)
	out = 2*tp/(2*tp + fp_fn)
	# Write your code here
	return round(out, 3)
