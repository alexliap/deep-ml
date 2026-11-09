def alloc_checks(running_gpu_config, profile):
    if profile["compute_slices"] > running_gpu_config["total_compute_slices"]:
        return False
    if profile["memory_gb"] > running_gpu_config["total_memory_gb"]:
        return False

    return True

def profile_check(profile, workload):
    if profile["compute_slices"] < workload["min_compute_slices"]:
        return False
    if profile["memory_gb"] < workload["min_memory_gb"]:
        return False
    
    return True

def update_gpu(running_gpu_config, profile):
    running_gpu_config["total_compute_slices"] -= profile["compute_slices"]
    running_gpu_config["total_memory_gb"] -= profile["memory_gb"]

    return running_gpu_config

def mig_resource_allocation(gpu_config: dict, mig_profiles: list[dict], workloads: list[dict]) -> dict:
    """
    Allocate MIG instances to workloads on a single GPU.
    
    Args:
        gpu_config: Dict with 'total_compute_slices' and 'total_memory_gb'
        mig_profiles: List of available MIG profile dicts
        workloads: List of workload requirement dicts
    
    Returns:
        Dict with allocation results and utilization metrics
    """
    workloads.sort(key=lambda p: (-p["min_compute_slices"], -p["min_memory_gb"]))

    running_gpu_config = gpu_config.copy()
    allocated = []
    rejected = []
    for work in workloads:
        for mig in mig_profiles:
            if not alloc_checks(running_gpu_config, mig):
                continue
            if not profile_check(mig, work):
                continue

            running_gpu_config = update_gpu(running_gpu_config, mig)

            allocated.append(
                {
                    "workload": work["name"],
                    "profile": mig["name"],
                    "compute_slices": mig["compute_slices"],
                    "memory_gb": mig["memory_gb"],
                }
            )

            break
        else:
            rejected.append(work["name"])
    
    total_compute_used = gpu_config["total_compute_slices"] - running_gpu_config["total_compute_slices"]
    total_memory_used = gpu_config["total_memory_gb"] - running_gpu_config["total_memory_gb"]
    compute_utilization = total_compute_used / gpu_config["total_compute_slices"]
    memory_utilization = total_memory_used / gpu_config["total_memory_gb"]
    workloads_served = len(allocated)
    workloads_rejected = len(workloads) - workloads_served

    return {
        "allocations": allocated,
        "total_compute_used": total_compute_used,
        "total_memory_used": total_memory_used,
        "compute_utilization": round(100*compute_utilization, 2),
        "memory_utilization": round(100*memory_utilization, 2),
        "workloads_served": workloads_served,
        "workloads_rejected": rejected,
    }

        
