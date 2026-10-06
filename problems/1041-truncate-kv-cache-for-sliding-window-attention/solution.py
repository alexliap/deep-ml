import numpy as np

def update_sliding_kv_cache(k_cache, v_cache, k_new, v_new, window_size):
    """
    Append new K/V to the cache and truncate to the last `window_size` tokens.

    Args:
        k_cache, v_cache: arrays of shape (B, H, S_cached, D)
        k_new, v_new:     arrays of shape (B, H, S_new, D)
        window_size:      int, maximum number of tokens to retain along seq axis

    Returns:
        Tuple (k_cache_updated, v_cache_updated), each of shape
        (B, H, min(S_cached + S_new, window_size), D).
    """
    if window_size <= 0:
        raise ValueError()

    k = np.concatenate((k_cache, k_new), axis=-2)
    v = np.concatenate((v_cache, v_new), axis=-2)
    effective_seqlen = min(k.shape[-2], window_size)

    return (k[:,:,-effective_seqlen:, :], v[:,:,-effective_seqlen:, :])
