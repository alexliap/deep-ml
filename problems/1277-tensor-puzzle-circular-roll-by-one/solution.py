import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    """Circular left shift by one."""
    out = np.zeros_like(a)
    out[:-1] = a[1:]
    out[-1] = a[0]
    return out
