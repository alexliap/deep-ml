import numpy as np

def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    """ 
    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
    
    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    B, S, _ = Q.shape
    D = Q.shape[-1] // num_heads
    group_size = num_heads // num_kv_heads

    # split into heads: (B, S, heads*D) -> (B, heads, S, D)
    q = Q.reshape(B, S, num_heads, D).transpose(0, 2, 1, 3)
    k = K.reshape(B, S, num_kv_heads, D).transpose(0, 2, 1, 3)
    v = V.reshape(B, S, num_kv_heads, D).transpose(0, 2, 1, 3)

    # repeat each KV head in place: [A, B] -> [A, A, B, B]
    k = np.repeat(k, group_size, axis=1)
    v = np.repeat(v, group_size, axis=1)

    # per-head attention: (B, heads, S, S)
    score = q @ k.transpose(0, 1, 3, 2) / np.sqrt(D)
    score = score - score.max(axis=-1, keepdims=True)
    weights = np.exp(score)
    weights = weights / weights.sum(axis=-1, keepdims=True)

    out = weights @ v                                # (B, heads, S, D)
    return out.transpose(0, 2, 1, 3).reshape(B, S, num_heads * D)