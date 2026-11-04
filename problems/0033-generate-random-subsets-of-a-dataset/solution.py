import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    subset_size = len(y) if replacements else len(y) // 2
    idx = np.array([
        np.random.choice(len(y), subset_size, replace=replacements)
        for _ in range(n_subsets)
    ])
    return [(X[i].tolist(), y[i].tolist()) for i in idx]