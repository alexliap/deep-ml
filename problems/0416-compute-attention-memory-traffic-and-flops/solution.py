def matmul_mem(B, M, K, P, bytes_per_element):
    read = (bytes_per_element * B * M * K) + (bytes_per_element * B * K * P)
    write = (bytes_per_element * B * M * P)
    return read + write

def matmul_flops(B, M, K, P):
    return 2 * M * K * P * B # 2MKP

def softmax_flops(B, seq_length, d_head):
    return (5 * d_head) * seq_length * B

def softmax_mem(B, M, K, bytes_per_element):
    read = bytes_per_element * B * M * K
    write = bytes_per_element * B * M * K
    return read + write

def attention_memory_flops(B: int, h: int, N: int, d: int, bytes_per_element: int = 2) -> dict:
    """
    Compute memory traffic and FLOPs for standard self-attention.

    Args:
        B: Batch size
        h: Number of attention heads
        N: Sequence length
        d: Head dimension
        bytes_per_element: Bytes per element (e.g., 2 for FP16, 4 for FP32)

    Returns:
        dict with keys:
            'qk_flops': int - FLOPs for Q @ K^T
            'softmax_flops': int - FLOPs for softmax
            'pv_flops': int - FLOPs for P @ V
            'total_flops': int - Total FLOPs
            'memory_bytes': int - Total memory traffic in bytes
            'arithmetic_intensity': float - FLOPs per byte, rounded to 2 decimal places
    """
    qk_flops = matmul_flops(B, N, d, N) * h
    qk_mem = matmul_mem(B, N, d, N, bytes_per_element) * h

    soft_flops = softmax_flops(B, N, N) * h
    soft_mem = softmax_mem(B, N, N, bytes_per_element) * h

    pv_flops = matmul_flops(B, N, N, d) * h
    pv_mem = matmul_mem(B, N, N, d, bytes_per_element) * h

    total_flops = qk_flops + pv_flops + soft_flops
    memory_bytes = qk_mem + pv_mem + soft_mem

    arithmetic_intensity = round(total_flops/memory_bytes, 2)

    return {
        "qk_flops": qk_flops,
        "softmax_flops": soft_flops,
        "pv_flops": pv_flops,
        "total_flops": total_flops,
        "memory_bytes": memory_bytes,
        "arithmetic_intensity": arithmetic_intensity
    }
    