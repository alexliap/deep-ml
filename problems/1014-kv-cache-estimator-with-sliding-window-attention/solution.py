def kv_cache_bytes(num_layers_full: int, num_layers_swa: int, batch: int, seqlen: int, window: int, num_kv_heads: int, head_dim: int, dtype_bytes: int) -> int:
    """
    Return total KV cache size in bytes across all layers.
    Full-attention layers use effective seqlen = seqlen.
    Sliding-window layers use effective seqlen = min(seqlen, window).
    Per layer: 2 * batch * eff_seqlen * num_kv_heads * head_dim * dtype_bytes.
    """
    # kv_cache_bytes = 0

    full_attn_bytes = 2 * batch * seqlen * num_kv_heads * head_dim * dtype_bytes
    full_attn_bytes *= num_layers_full

    swa_attn_bytes = 2 * batch * min(seqlen, window) * num_kv_heads * head_dim * dtype_bytes
    swa_attn_bytes *= num_layers_swa

    return swa_attn_bytes + full_attn_bytes