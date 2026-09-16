import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    predicted_probs = np.clip(predicted_probs, a_min=epsilon, a_max=1e5)
    ce_loss = -np.sum(true_labels * np.log(predicted_probs), axis=1)
    return np.mean(ce_loss)
