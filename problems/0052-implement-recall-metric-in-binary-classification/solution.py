import numpy as np

def recall(y_true, y_pred):
    tp = 0
    fn = 0

    for i in range(len(y_pred)):
        if y_pred[i] + y_true[i] == 2:
            tp += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn += 1

    return tp / (tp + fn) if (tp + fn) > 0 else 0
