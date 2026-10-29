# --- cell 1 ---
import numpy as np

def normalize(x):
    '''
    Apply normalization to a batch of feature vectors.
    
    Args:
        x: numpy array of shape (batch_size, features)
           Raw activations from a neural network layer
    
    Returns:
        numpy array of same shape (batch_size, features)
        Normalized activations
    
    Requirements:
        - Must not be the identity function
        - Must work on 2D arrays of any size
        - Must be deterministic
        - Must not produce NaN or Inf
        - Normalize across the LAST dimension (features)
    
    Hint: Modern LLMs use RMSNorm, which normalizes by the
    root mean square of the feature values. But you can
    implement any normalization you like!
    '''
    # rms norm
    x = x.astype(np.float32)
    eps = 1e-6
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    result = 2 * x / (rms)
    
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)