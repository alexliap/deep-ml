import numpy as np

def softplus(z: float):
    return np.log(1 + np.exp(z))

def sigmoid(z):
    return 1/(1+np.exp(-np.array(z)))

def gated_deltanet(q, k, v, a, b, g, A_log, rms_weight, eps=1e-6):
    """
    Args:
        q, k, v: lists of shape (T, d)
        a, b: lists of shape (T,) -- pre-activation gate logits
        g: list of shape (T, d) -- output gate logits
        A_log: float -- learned log time-scale
        rms_weight: list of shape (d,) -- RMSNorm scale
        eps: float -- numerical stability constant

    Returns:
        Nested list of shape (T, d), rounded to 4 decimals.
    """
    q_norm_denom = np.sqrt(np.sum(np.power(q, 2), axis=-1, keepdims=True)) + eps
    q_norm = q / q_norm_denom

    k_norm_denom = np.sqrt(np.sum(np.power(k, 2), axis=-1, keepdims=True)) + eps
    k_norm = k / k_norm_denom

    alpha_t = np.exp(-softplus(a) * np.exp(A_log))
    beta_t = sigmoid(b)

    T, d_model = np.array(q).shape
    S = np.zeros((d_model, d_model))
    y_t = []
    for i in range(T):
        S = alpha_t[i] * S
        delta = (v[i] - np.matmul(k_norm[i], S.T)) * beta_t[i]
        S = S + np.outer(delta, k_norm[i])
        o_t = np.matmul(q_norm[i], S.T)

        denom = np.sqrt(np.mean(o_t**2, axis=-1, keepdims=True) + eps)
        o_norm = (o_t / denom) * rms_weight

        y_t.append(o_norm * (g[i] * sigmoid(g[i])))
        
    return np.round(y_t, 4).tolist()

