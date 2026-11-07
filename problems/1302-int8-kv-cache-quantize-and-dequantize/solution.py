import numpy as np

def quantize_kv_int8(x):
    """
    Symmetric INT8 quantize along the last axis.

    Returns (q, scale, x_hat).
    """
    s = np.max(np.abs(x), axis=-1, keepdims=True) / 127
    s = np.where(s < 1e-12, 1.0, s)
    q = np.clip(np.rint(x/s), -127, 127).astype(np.int8)
    x_hat = q * s

    return (q, s, x_hat)
