import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    exp_logtis = np.exp(logits - np.max(logits))
    probs = exp_logtis/np.sum(exp_logtis)
    quantile = np.quantile(probs, p)

    order_idxs = np.argsort(-probs)

    cum_sum = np.cumsum(probs[order_idxs])

    eligible_idxs = np.sum(np.cumsum(probs[order_idxs]) < p) + 1
    keep_idxs = order_idxs[:eligible_idxs]
    zero_idxs = order_idxs[eligible_idxs:]

    prob_sum = np.sum(probs[keep_idxs])

    min_prob = np.min(probs[keep_idxs])
    # probs[probs < min_prob] = 0
    probs[zero_idxs] = 0

    return np.round(probs/prob_sum, 4)
