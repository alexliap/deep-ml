import numpy as np

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    qk = np.matmul(Q, K.T)/np.sqrt(K.shape[1])
    for i in range(qk.shape[1]):
        # mask using slicing
        start = i - window_size
        end = i + window_size + 1
        if start < 0:
            start = 0
        
        qk[:start, i] = - np.inf
        qk[end:, i] = - np.inf

    soft_nom = np.exp(qk - np.max(qk, axis=-1, keepdims=True))
    softmax = soft_nom / np.sum(soft_nom, axis=-1, keepdims=True)

    return np.matmul(softmax, V)