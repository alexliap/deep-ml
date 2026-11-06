def roofline_analysis(peak_gflops: float, peak_bandwidth_gbs: float, operations: list) -> dict:
	"""
	Perform Roofline Model analysis for GPU operations.

	Args:
		peak_gflops: Peak compute throughput in GFLOPS
		peak_bandwidth_gbs: Peak memory bandwidth in GB/s
		operations: List of dicts with keys 'name', 'flops', 'bytes'

	Returns:
		Dict with 'ridge_point' and 'operations' list containing
		per-operation analysis results.
	"""
	ops_byte_ratio = peak_gflops/peak_bandwidth_gbs
	out = {
		"ridge_point": ops_byte_ratio,
		"operations": []
	}
	for op in operations:
		intensity = op["flops"] / op["bytes"]
		out["operations"].append({
			"name": op["name"],
			"operational_intensity": intensity,
			"attainable_gflops": min(peak_gflops, peak_bandwidth_gbs*intensity),
			"bottleneck": 'compute-bound' if intensity > ops_byte_ratio else 'memory-bound',
			"efficiency": 100*min(peak_gflops, peak_bandwidth_gbs*intensity)/peak_gflops
		})

	return out