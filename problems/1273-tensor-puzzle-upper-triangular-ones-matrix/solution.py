import numpy as np

def triu_ones(n: int) -> np.ndarray:
    """n x n upper-triangular matrix of ones (including diagonal)."""
    a = np.arange(n).reshape(1, -1)
    b = np.arange(n).reshape(-1, 1)
    out = a >= b

    return np.float32(out)
