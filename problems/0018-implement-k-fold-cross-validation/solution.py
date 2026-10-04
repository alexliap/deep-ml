import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    split_size = n_samples // k
    first_split_size = split_size + n_samples % k

    sizes = [0] + [first_split_size] + [split_size] * (k-1)

    k_fold = []
    offset = 0
    for i in range(len(sizes)-1):
        offset += sizes[i]
        start = offset
        end = offset + sizes[i+1]
        test = indices[start:end]
        train = np.concatenate((indices[:start], indices[end:]))

        k_fold.append((train, test))

    return k_fold