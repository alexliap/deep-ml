import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""
    return np.array([A[i, i] for i in range(A.shape[0])])
