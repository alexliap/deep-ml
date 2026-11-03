import numpy as np

def layernorm(x, gamma, beta, eps):
    mean_x = np.mean(x, axis=-1, keepdims=True)
    var_x = np.var(x, axis=-1, keepdims=True)

    return beta + gamma * (x - mean_x) / np.sqrt(var_x + eps)

def softmax(x):
    soft_nom = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return soft_nom / np.sum(soft_nom, axis=-1, keepdims=True)

def linear(x, W, b):
    return x @ W + b

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    # attention
    q = np.split(X @ weights["W_q"], num_heads, axis=-1)
    k = np.split(X @ weights["W_k"], num_heads, axis=-1)
    k = np.transpose(k, (0, 1, 3, 2))
    v = np.split(X @ weights["W_v"], num_heads, axis=-1)

    outs = []
    for i in range(num_heads):
        scores = np.matmul(q[i], k[i])/np.sqrt(X.shape[-1])
        soft_scores = softmax(scores)
        out = soft_scores @ v[i]
        outs.append(out)

    attn  = np.concatenate(outs, axis=-1) @ weights["W_o"]

    z = layernorm(X + attn, 
                    weights["gamma1"], 
                    weights["beta1"], eps)

    # ffn
    lin1 = linear(z, weights["W1"], weights["b1"])
    lin1 = np.maximum(0, lin1)
    lin2 = linear(lin1, weights["W2"], weights["b2"])
    out = layernorm(z + lin2, weights["gamma2"], weights["beta2"], eps)

    return out