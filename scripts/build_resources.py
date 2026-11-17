#!/usr/bin/env python3
"""Conservative CUDA build parallelism, respecting CPU and memory limits."""
import json
import os
from pathlib import Path

GIB = 1024**3


def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def choose_jobs(cpus: int, available_memory: int) -> int:
    # CUTLASS compilation has reached ~18 GiB per process in this project.
    # Budget 24 GiB/job and leave a quarter of available memory uncommitted.
    return max(1, min(8, cpus, available_memory * 3 // 4 // (24 * GIB)))


def resources():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    # Unlike nproc, affinity is not reduced by OMP_NUM_THREADS used by Torch.
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            cpus = min(cpus, max(1, int(quota)//int(period)))
    except (OSError, ValueError):
        quota = read_int('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        period = read_int('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if quota is not None and quota > 0 and period:
            cpus = min(cpus, max(1, quota//period))
    try:
        memory = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    except (OSError, StopIteration, ValueError):
        memory = 32 * GIB
    for limit_path, usage_path in (
        ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes'),
    ):
        limit, usage = read_int(limit_path), read_int(usage_path)
        if limit is not None and usage is not None:
            memory = min(memory, max(0, limit-usage))
    return dict(cpus=cpus,available_memory_bytes=memory,jobs=choose_jobs(cpus,memory))


if __name__ == '__main__':
    print(json.dumps(resources(),indent=2))
