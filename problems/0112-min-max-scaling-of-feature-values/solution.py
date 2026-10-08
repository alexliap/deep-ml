def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    import numpy as np

    out = (np.array(x) - np.min(x))/(np.max(x) - np.min(x))
    return out.tolist()