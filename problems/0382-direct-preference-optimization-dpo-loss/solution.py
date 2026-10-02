import numpy as np

def dpo_loss(log_probs_chosen_policy: list, log_probs_rejected_policy: list,
            log_probs_chosen_ref: list, log_probs_rejected_ref: list,
            beta: float) -> dict:
    """
    Compute the Direct Preference Optimization (DPO) loss.
    """
    logits_acc = beta * (np.array(log_probs_chosen_policy) - np.array(log_probs_chosen_ref))

    logits_rej = beta * (np.array(log_probs_rejected_policy) - np.array(log_probs_rejected_ref))

    loss = -np.log(1/(1 + np.exp(-(logits_acc - logits_rej))))

    return {
        "loss": np.round(np.mean(loss), 4), 
        "chosen_rewards": np.round(logits_acc, 4).tolist(), 
        "rejected_rewards": np.round(logits_rej, 4).tolist()
    }