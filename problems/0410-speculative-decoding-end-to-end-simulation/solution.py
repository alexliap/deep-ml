import numpy as np

def speculative_decode(draft_probs: list, target_probs: list, draft_tokens: list, accept_rand: list, sample_rand: list) -> list:
    """
    Simulate speculative decoding end-to-end.
    
    Args:
        draft_probs: K arrays of shape (vocab_size,) - draft model probability distributions
        target_probs: K+1 arrays of shape (vocab_size,) - target model probability distributions
        draft_tokens: K integers - token indices proposed by draft model
        accept_rand: K floats in [0,1) - random values for acceptance decisions
        sample_rand: K+1 floats in [0,1) - random values for token sampling
    """
    draft_probs = np.array(draft_probs)
    target_probs = np.array(target_probs)
    draft_tokens = np.array(draft_tokens)
    accept_rand = np.array(accept_rand)
    sample_rand = np.array(sample_rand)

    acc_ratio = np.diag(target_probs[:-1, draft_tokens] / draft_probs[:, draft_tokens])

    acc_token_len = 0
    for i in range(len(acc_ratio)):
        if acc_ratio[i] > accept_rand[i]:
            acc_token_len += 1
        else:
            break

    accepted_tokens = []
    if acc_token_len == len(draft_tokens):
        accepted_tokens.extend(draft_tokens) # correct

        cdf = np.cumsum(target_probs[acc_token_len])
        k_plus_1 = np.where(cdf > sample_rand[acc_token_len])[0][0]
        accepted_tokens.extend([k_plus_1.item()])
    else:
        accepted_tokens.extend(draft_tokens[:acc_token_len])

        idx = acc_token_len
        adj_dist = np.maximum((target_probs[idx] - draft_probs[idx]), 0)
        norm_adj_dist = adj_dist / np.sum(adj_dist)

        cdf = np.cumsum(norm_adj_dist)
        k_plus_1 = np.where(cdf > sample_rand[acc_token_len])[0][0]
        
        accepted_tokens.extend([k_plus_1.item()])

    return accepted_tokens
