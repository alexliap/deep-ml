import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    q = np.matmul(X, W_q)
    k = np.matmul(X, W_k)
    v = np.matmul(X, W_v)

    return q, k, v

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    score = np.matmul(Q, K.T)/np.sqrt(K.shape[1])
    soft_nominator = np.exp(score - np.max(score, axis=-1, keepdims=True))
    softmax = soft_nominator / np.sum(soft_nominator, axis=-1, keepdims=True)

    return np.matmul(softmax, V)


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # return self_attention(Q, K, V)
    q = np.split(Q, n_heads, axis=1) # seq_len, d
    k = np.split(K, n_heads, axis=1)
    v = np.split(V, n_heads, axis=1)

    out = None
    for i in range(n_heads):
        if out is None:
            out = self_attention(q[i], k[i], v[i])
        else:
            out = np.hstack([out, self_attention(q[i], k[i], v[i])])
    return out