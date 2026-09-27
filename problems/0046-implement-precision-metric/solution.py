import numpy as np
def precision(y_true, y_pred):
	tp = 0
	fp = 0
	for i in range(len(y_pred)):
		if y_pred[i] == 1:
			match y_true[i]:
				case 0:
					fp += 1
				case 1:
					tp += 1

	return tp/(tp+fp) if (tp+fp) > 0 else 0
