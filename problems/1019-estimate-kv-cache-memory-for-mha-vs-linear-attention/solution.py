def kv_cache_memory(batch_size: int, seq_len: int, embed_dim: int, n_heads: int, n_layers: int, bytes_per_elem: int) -> list:
    """
    Return [mha_bytes, linear_bytes, mixed_bytes] for the three architectures.
    Mixed uses a 3:1 ratio of linear-attention layers to MHA layers.
    """
    d_head = embed_dim // n_heads
    mha_bytes = 2 * batch_size * seq_len * n_heads * d_head * bytes_per_elem

    linear_bytes = batch_size * n_heads * d_head * d_head * bytes_per_elem

    blocks = n_layers // 4
    mixed_bytes = blocks * 3 * linear_bytes + blocks * 1 * mha_bytes

    return [n_layers * mha_bytes, n_layers * linear_bytes, mixed_bytes]