import numpy as np

def eye(n: int) -> np.ndarray:
    """n x n identity matrix without np.eye."""
    a = np.arange(n).reshape(1, -1)
    b = np.arange(n).reshape(-1, 1)
    out = a == b

    return np.float32(out)
