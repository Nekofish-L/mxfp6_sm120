"""Pure W6A8 search policy, independent of CUDA, cache I/O and environment.

Keep ordering stable: it affects seeded shuffling and ties in measured ranks.
Resource-aware shortlisting can be evaluated here without changing launch code.
"""
from dataclasses import dataclass
import math

from ._w6a8_registry import KERNEL_NAMES, STREAM_K_CONFIG_IDS

FALLBACK_CONFIG_ID = -1
RASTER_NAMES = ("heuristic", "along_m", "along_n")
SWIZZLES = (1, 2, 4, 8)


@dataclass(frozen=True)
class W6A8Config:
    config_id: int
    swizzle: int
    raster_order: int

    @property
    def kernel(self) -> str:
        if self.config_id == FALLBACK_CONFIG_ID:
            return "builtin_fallback"
        return KERNEL_NAMES[self.config_id]

    @property
    def raster(self) -> str:
        return RASTER_NAMES[self.raster_order]


@dataclass(frozen=True)
class AutotuneResult:
    config: W6A8Config
    latency_us: float
    fallback_us: float
    runner_up_us: float
    samples: int


def candidate_kernel_ids(
    m: int, k: int, *, all_families_max_m: int, stream_k_enabled: bool
) -> tuple[int, ...]:
    # Every AOT kernel family is geometrically valid for any M, and M-regime
    # shortlists provably excluded winners through M=512 (notably swapped
    # 64x32x128-stage3 at n=5120, k=3072). Above that point the observed
    # family gaps are small enough for finite-sample ranking noise to dominate
    # the larger search. Keep the stable normal-family shortlist there; an
    # offline run can still opt into every family by raising the threshold.
    if m <= all_families_max_m:
        kernels = list(range(len(KERNEL_NAMES)))
    elif m <= 16:
        kernels = [0, 1, 2, 3, 4, 5, 29]
    elif m <= 32:
        kernels = [
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        ]
    else:
        kernels = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    if k < 1024:
        kernels = [
            kernel for kernel in kernels
            if kernel not in STREAM_K_CONFIG_IDS
        ]
    if not stream_k_enabled:
        kernels = [
            kernel for kernel in kernels
            if kernel not in STREAM_K_CONFIG_IDS
        ]
    return tuple(kernels)


def refinement_kernel_ids(
    coarse_results: list[tuple[float, W6A8Config]],
    kernel_ids: list[int],
    *,
    exhaustive: bool,
) -> list[int]:
    best_by_kernel: dict[int, float] = {}
    for latency, config in coarse_results:
        best_by_kernel[config.config_id] = min(
            latency, best_by_kernel.get(config.config_id, math.inf)
        )
    ranked_kernel_ids = [
        kernel_id
        for kernel_id, _ in sorted(
            best_by_kernel.items(), key=lambda item: item[1]
        )
    ]
    top_kernel_ids = ranked_kernel_ids if exhaustive else ranked_kernel_ids[:3]
    return top_kernel_ids if top_kernel_ids else kernel_ids[:3]
