def calculate_inference_stats(latencies_ms: list) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: list of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """
    def interpolate(latencies_ms, p):
        latencies_ms.sort()

        k = (p / 100) * (len(latencies_ms) - 1)
        lo = int(k)
        hi = int(k+1) if len(latencies_ms) > 1 else lo
        if lo == hi:
            return latencies_ms[lo]

        frac = k - lo
        return latencies_ms[lo] + frac * (latencies_ms[hi] - latencies_ms[lo])

    if latencies_ms:
        avg_latency_ms = sum(latencies_ms)/len(latencies_ms)
        throughput_per_sec = 1000/avg_latency_ms

        p50_ms = interpolate(latencies_ms, 50)
        p95_ms = interpolate(latencies_ms, 95)
        p99_ms = interpolate(latencies_ms, 99)
        
        return {"throughput_per_sec": round(throughput_per_sec, 2),
                "avg_latency_ms": round(avg_latency_ms, 2),
                "p50_ms": round(p50_ms, 2), 
                "p95_ms": round(p95_ms, 2), 
                "p99_ms": round(p99_ms, 2)}
    else:
        return {}