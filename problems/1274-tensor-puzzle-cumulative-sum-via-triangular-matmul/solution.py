import numpy as np

def cumsum(a: np.ndarray) -> np.ndarray:
    """Inclusive cumulative sum of 1-D array a without np.cumsum."""
    x1 = np.arange(a.shape[0]).reshape(1, -1)
    x2 = np.arange(a.shape[0]).reshape(-1, 1)
    tril = np.float32(x1 >= x2)

    return a @ tril
