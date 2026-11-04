import torch
import torch.nn.functional as F

def noisy_topk_gating(
    X: torch.Tensor,
    W_g: torch.Tensor,
    W_noise: torch.Tensor,
    N: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    h_base = X @ W_g
    h_noise = X @ W_noise
    h = h_base + N * F.softplus(h_noise)
    h_topk, indices = torch.topk(h, k, dim=-1)

    result = torch.full_like(h, - torch.inf)
    result.scatter_(dim=-1, index=indices, src=h_topk)

    # h[~indices] = - torch.inf
    out = F.softmax(result, dim=-1)

    return out