import numpy as np

def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product of 1-D arrays a and b via broadcasting."""
    a = a.reshape(-1, 1)
    b = b.reshape(1, -1)
    return a * b
