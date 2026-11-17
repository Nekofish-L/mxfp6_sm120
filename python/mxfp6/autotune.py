"""First-use native W6A8 autotuning with a persistent dispatch cache.

The kernels themselves are AOT-compiled into ``mxfp6_torch``.  For a shape
without a checked-in exact override, this module benchmarks an M-aware
candidate policy once, validates the winner, installs it in the C++ dispatcher,
and persists the decision for later processes.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import random
import threading

import torch

from . import _autotune_cache
from ._autotune_cache import cache_key as _cache_key
from ._loader import load_library
from ._autotune_benchmark import measure_graph
from ._autotune_policy import (
    FALLBACK_CONFIG_ID, RASTER_NAMES, SWIZZLES, AutotuneResult, W6A8Config,
    candidate_kernel_ids, refinement_kernel_ids,
)
from ._w6a8_registry import CANDIDATE_ABI, KERNEL_NAMES, STREAM_K_CONFIG_IDS


AUTOTUNE_SCHEMA = 1
CANDIDATE_POLICY = "hybrid-all-families-through-m512-v2"
TIMING_POLICY = "gemm-cuda-graph-events-two-stage-v5"
DEFAULT_AUTOTUNE_WARMUP = 3
DEFAULT_AUTOTUNE_ITERATIONS = 10
DEFAULT_AUTOTUNE_REPEATS = 3
DEFAULT_ALL_FAMILIES_MAX_M = 512


_STATE_LOCK = threading.Lock()
_KEY_LOCKS: dict[str, threading.Lock] = {}
_DECISIONS: dict[tuple[int, str], W6A8Config] = {}
_LIBRARY_FINGERPRINT: str | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def is_autotune_enabled() -> bool:
    """Return whether the autotune cache/tuner is enabled."""
    return autotune_mode() != "off"


def autotune_mode() -> str:
    """Return ``on``, ``off``, or deployment-safe ``cache_only`` mode."""
    raw = os.environ.get("MXFP6_AUTOTUNE")
    if raw is None:
        return "on"
    value = raw.strip().lower()
    if value == "cache_only":
        return "cache_only"
    if value in ("", "0", "false", "no", "off"):
        return "off"
    return "on"


def is_autotune_cache_only() -> bool:
    """Return whether misses may read cache entries but must not profile."""
    return autotune_mode() == "cache_only"


def is_stream_k_enabled() -> bool:
    """Return whether Stream-K kernels are enabled (enabled by default)."""
    raw = os.environ.get("MXFP6_STREAM_K")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false")


def should_tune_exact_shapes() -> bool:
    """Return whether checked-in exact dispatch entries should be retuned."""
    return _env_bool("MXFP6_AUTOTUNE_EXACT", False)


def _use_exhaustive_search() -> bool:
    """Return whether the refinement pass should retain every kernel family."""
    return _env_bool("MXFP6_AUTOTUNE_EXHAUSTIVE", False)


def _verbose(message: str) -> None:
    if _env_bool("MXFP6_AUTOTUNE_VERBOSE", False):
        print(f"[mxfp6 autotune] {message}", flush=True)


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


def _nonnegative_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative; got {value}")
    return value


def _all_families_max_m() -> int:
    """Return the largest M that receives full-family coarse search."""
    return _nonnegative_env_int(
        "MXFP6_AUTOTUNE_ALL_FAMILIES_MAX_M",
        DEFAULT_ALL_FAMILIES_MAX_M,
    )


def _output_dtype_name(out_dtype: torch.dtype) -> str:
    if out_dtype == torch.float16:
        return "fp16"
    if out_dtype == torch.bfloat16:
        return "bf16"
    raise TypeError(
        "out_dtype must be torch.float16 or torch.bfloat16; "
        f"got {out_dtype}"
    )


def _cache_root() -> Path:
    override = os.environ.get("MXFP6_AUTOTUNE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "mxfp6" / "autotune"


def _library_fingerprint() -> str:
    global _LIBRARY_FINGERPRINT
    if _LIBRARY_FINGERPRINT is not None:
        return _LIBRARY_FINGERPRINT
    library = load_library()
    digest = hashlib.sha256(CANDIDATE_ABI.encode())
    with library.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    _LIBRARY_FINGERPRINT = digest.hexdigest()
    return _LIBRARY_FINGERPRINT


def _device_descriptor(device_index: int) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "name": properties.name,
        "sm": [properties.major, properties.minor],
        "sm_count": properties.multi_processor_count,
        "l2_bytes": getattr(properties, "L2_cache_size", 0),
        "memory_bus_width": getattr(properties, "memory_bus_width", 0),
        "total_memory": properties.total_memory,
        "torch_cuda": torch.version.cuda,
    }


def _descriptor(
    device_index: int,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> dict[str, object]:
    flush_l2_mb = _nonnegative_env_int(
        "MXFP6_AUTOTUNE_FLUSH_L2_MB", 0
    )
    return {
        "schema": AUTOTUNE_SCHEMA,
        "candidate_abi": CANDIDATE_ABI,
        # The candidate set changed independently of the native ABI; fold the
        # policy into the cache key so decisions tuned under another M regime
        # are retuned rather than replayed.
        "candidate_policy": {
            "name": CANDIDATE_POLICY,
            "all_families_max_m": _all_families_max_m(),
        },
        "timing_policy": TIMING_POLICY,
        "library": _library_fingerprint(),
        "stream_k_enabled": is_stream_k_enabled(),
        "device": _device_descriptor(device_index),
        "problem": {
            "m": m,
            "n": n,
            "k": k,
            "layout": "NT",
            "a": "mxfp8_e4m3_ue8m0_group32",
            "b": "packed_mxfp6_e3m2_ue8m0_group32",
            "accumulator": "fp32",
            "output": _output_dtype_name(out_dtype),
        },
        "measurement": {
            "cache_regime": "warm" if flush_l2_mb == 0 else "explicit_flush",
            "search_mode": (
                "exhaustive" if _use_exhaustive_search() else "two_stage"
            ),
            "flush_l2_mb": flush_l2_mb,
            "warmup": _positive_env_int(
                "MXFP6_AUTOTUNE_WARMUP", DEFAULT_AUTOTUNE_WARMUP
            ),
            "iterations": _positive_env_int(
                "MXFP6_AUTOTUNE_ITERATIONS", DEFAULT_AUTOTUNE_ITERATIONS
            ),
            "repeats": _positive_env_int(
                "MXFP6_AUTOTUNE_REPEATS", DEFAULT_AUTOTUNE_REPEATS
            ),
            "minimum_improvement": float(
                os.environ.get("MXFP6_AUTOTUNE_MIN_IMPROVEMENT", "0.02")
            ),
        },
    }


def _get_key_lock(key: str) -> threading.Lock:
    with _STATE_LOCK:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


def _file_lock(key: str):
    return _autotune_cache.file_lock(_cache_root(), key)


def _read_entry(key: str, descriptor: dict[str, object]) -> W6A8Config | None:
    return _autotune_cache.read_entry(_cache_root(), key, descriptor)


def _write_entry(key: str, descriptor: dict[str, object], result: AutotuneResult) -> None:
    _autotune_cache.write_entry(_cache_root(), key, descriptor, result)


def _is_compiling_or_capturing() -> bool:
    try:
        if torch.cuda.is_current_stream_capturing():
            return True
    except RuntimeError:
        pass
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    return bool(is_compiling is not None and is_compiling())


def can_autotune_now() -> bool:
    """Return whether synchronization and cache I/O are safe right now."""
    return not _is_compiling_or_capturing()


def _kernel_ids(m: int, k: int) -> tuple[int, ...]:
    return candidate_kernel_ids(
        m, k, all_families_max_m=_all_families_max_m(),
        stream_k_enabled=is_stream_k_enabled(),
    )


def _run_config(
    config: W6A8Config,
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if config.config_id == FALLBACK_CONFIG_ID:
        return torch.ops.mxfp6.gemm_w6a8(
            a, b, sfa, sfb, m, n, k, 1.0, out_dtype
        )
    return torch.ops.mxfp6.gemm_w6a8_config(
        a,
        b,
        sfa,
        sfb,
        m,
        n,
        k,
        1.0,
        config.config_id,
        config.swizzle,
        config.raster_order,
        out_dtype,
    )


def _measure_config(
    config: W6A8Config,
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
    warmup: int,
    iterations: int,
    repeats: int,
    flush: torch.Tensor | None,
) -> float:
    return measure_graph(
        lambda: _run_config(config, a, b, sfa, sfb, m, n, k, out_dtype),
        a.device, warmup, iterations, repeats, flush,
    )


def _candidate_configs(m: int, k: int) -> tuple[list[W6A8Config], list[int]]:
    kernel_ids = _kernel_ids(m, k)
    coarse = [
        W6A8Config(kernel_id, swizzle, 0)
        for kernel_id in kernel_ids
        for swizzle in SWIZZLES
    ]
    return coarse, list(kernel_ids)


def _autotune_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> AutotuneResult:
    warmup = _positive_env_int(
        "MXFP6_AUTOTUNE_WARMUP", DEFAULT_AUTOTUNE_WARMUP
    )
    iterations = _positive_env_int(
        "MXFP6_AUTOTUNE_ITERATIONS", DEFAULT_AUTOTUNE_ITERATIONS
    )
    repeats = _positive_env_int(
        "MXFP6_AUTOTUNE_REPEATS", DEFAULT_AUTOTUNE_REPEATS
    )
    flush_mb = _nonnegative_env_int("MXFP6_AUTOTUNE_FLUSH_L2_MB", 0)
    flush = None
    if flush_mb:
        flush = torch.empty(
            flush_mb * 1_000_000 // 4,
            dtype=torch.int32,
            device=a.device,
        )

    # Ensure the reference call cannot accidentally use a stale in-process
    # override while this exact shape is being retuned.
    torch.ops.mxfp6.set_w6a8_config(
        a, m, n, k, -1, 1, 0, out_dtype
    )
    fallback = W6A8Config(FALLBACK_CONFIG_ID, 1, 0)
    fallback_output = _run_config(
        fallback, a, b, sfa, sfb, m, n, k, out_dtype
    )
    fallback_us = _measure_config(
        fallback,
        a,
        b,
        sfa,
        sfb,
        m,
        n,
        k,
        out_dtype,
        warmup,
        iterations,
        repeats,
        flush,
    )

    coarse, kernel_ids = _candidate_configs(m, k)
    random.Random((m << 42) ^ (n << 21) ^ k).shuffle(coarse)
    coarse_results: list[tuple[float, W6A8Config]] = []
    for config in coarse:
        try:
            latency = _measure_config(
                config,
                a,
                b,
                sfa,
                sfb,
                m,
                n,
                k,
                out_dtype,
                warmup,
                iterations,
                1,
                flush,
            )
        except RuntimeError:
            continue
        if math.isfinite(latency) and latency > 0.0:
            coarse_results.append((latency, config))

    top_kernel_ids = refinement_kernel_ids(
        coarse_results, kernel_ids, exhaustive=_use_exhaustive_search(),
    )

    refined = [
        W6A8Config(kernel_id, swizzle, raster)
        for kernel_id in top_kernel_ids
        for swizzle in SWIZZLES
        for raster in range(len(RASTER_NAMES))
    ]
    random.Random((k << 42) ^ (n << 21) ^ m).shuffle(refined)
    refined_results: list[tuple[float, W6A8Config]] = []
    for config in refined:
        try:
            latency = _measure_config(
                config,
                a,
                b,
                sfa,
                sfb,
                m,
                n,
                k,
                out_dtype,
                warmup,
                iterations,
                repeats,
                flush,
            )
        except RuntimeError:
            continue
        if math.isfinite(latency) and latency > 0.0:
            refined_results.append((latency, config))
    refined_results.sort(key=lambda item: item[0])

    selected = fallback
    selected_us = fallback_us
    valid_results: list[tuple[float, W6A8Config]] = []
    for latency, config in refined_results:
        try:
            candidate_output = _run_config(
                config, a, b, sfa, sfb, m, n, k, out_dtype
            )
            torch.testing.assert_close(
                candidate_output,
                fallback_output,
                rtol=2.0e-3,
                atol=0.5,
            )
        except (AssertionError, RuntimeError):
            continue
        valid_results.append((latency, config))

    minimum_improvement = float(
        os.environ.get("MXFP6_AUTOTUNE_MIN_IMPROVEMENT", "0.02")
    )
    if not 0.0 <= minimum_improvement < 1.0:
        raise ValueError(
            "MXFP6_AUTOTUNE_MIN_IMPROVEMENT must be in [0,1)"
        )
    if valid_results and valid_results[0][0] < fallback_us * (
        1.0 - minimum_improvement
    ):
        selected_us, selected = valid_results[0]

    all_latencies = sorted(
        [fallback_us] + [latency for latency, _ in valid_results]
    )
    runner_up_us = (
        all_latencies[1] if len(all_latencies) > 1 else fallback_us
    )
    return AutotuneResult(
        selected,
        selected_us,
        fallback_us,
        runner_up_us,
        repeats,
    )


def _autotune(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
) -> AutotuneResult:
    # Candidate and fallback probes are not production selections. Suppress
    # planner collection until the selected override is installed and the
    # caller performs its normal warmup launch.
    previous = torch.ops.mxfp6._set_workspace_collection(a, False)
    try:
        return _autotune_impl(
            a, b, sfa, sfb, m, n, k, out_dtype
        )
    finally:
        torch.ops.mxfp6._set_workspace_collection(a, previous)


def _install(
    anchor: torch.Tensor,
    m: int,
    n: int,
    k: int,
    config: W6A8Config,
    out_dtype: torch.dtype,
) -> None:
    if config.config_id == FALLBACK_CONFIG_ID:
        torch.ops.mxfp6.set_w6a8_config(
            anchor, m, n, k, -1, 1, 0, out_dtype
        )
        return
    torch.ops.mxfp6.set_w6a8_config(
        anchor,
        m,
        n,
        k,
        config.config_id,
        config.swizzle,
        config.raster_order,
        out_dtype,
    )


def ensure_w6a8_tuned(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    m: int,
    n: int,
    k: int,
    *,
    out_dtype: torch.dtype = torch.float16,
    force: bool = False,
) -> W6A8Config | None:
    """Install a cached/native winner, tuning this shape once on a miss.

    In ``cache_only`` mode, a miss returns ``None`` without profiling so the
    caller retains the native static dispatcher; this remains true even when
    ``force=True``. Returns ``None`` when autotuning is disabled or unsafe in
    the current capture/compile context. A fallback decision is represented by
    config ID ``-1`` and is also cached, preventing repeated unsuccessful
    searches.
    """
    if not force and not is_autotune_enabled():
        return None
    if _is_compiling_or_capturing():
        return None
    cache_only = is_autotune_cache_only()
    cache_lookup = not force or cache_only
    device_index = a.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    extension_abi = torch.ops.mxfp6.w6a8_config_abi(a)
    if extension_abi != CANDIDATE_ABI:
        raise RuntimeError(
            "Python/native W6A8 autotune ABI mismatch: "
            f"Python expects {CANDIDATE_ABI!r}, extension reports "
            f"{extension_abi!r}"
        )
    _output_dtype_name(out_dtype)
    descriptor = _descriptor(device_index, m, n, k, out_dtype)
    key = _cache_key(descriptor)
    process_key = (device_index, key)
    if cache_lookup:
        with _STATE_LOCK:
            existing = _DECISIONS.get(process_key)
        if existing is not None:
            return existing

    with _get_key_lock(key):
        if cache_lookup:
            with _STATE_LOCK:
                existing = _DECISIONS.get(process_key)
            if existing is not None:
                return existing

        cached = _read_entry(key, descriptor) if cache_lookup else None
        if cached is not None:
            _install(a, m, n, k, cached, out_dtype)
            with _STATE_LOCK:
                _DECISIONS[process_key] = cached
            _verbose(
                f"cache hit {m}x{n}x{k}: {cached.kernel}, "
                f"{cached.raster}/sw{cached.swizzle}"
            )
            return cached

        if cache_only:
            _verbose(
                f"cache miss {m}x{n}x{k}; using native static dispatcher"
            )
            return None

        result: AutotuneResult | None = None
        try:
            with _file_lock(key):
                cached = None if force else _read_entry(key, descriptor)
                if cached is not None:
                    result_config = cached
                else:
                    _verbose(
                        f"tuning {m}x{n}x{k} on CUDA device {device_index}"
                    )
                    result = _autotune(
                        a, b, sfa, sfb, m, n, k, out_dtype
                    )
                    result_config = result.config
                    _write_entry(key, descriptor, result)
        except OSError as error:
            # A read-only home directory must not make the GEMM unusable. Keep
            # an in-process decision even when persistence is unavailable.
            _verbose(f"persistent cache unavailable: {error}")
            if result is None:
                result = _autotune(
                    a, b, sfa, sfb, m, n, k, out_dtype
                )
            result_config = result.config
        except RuntimeError as error:
            # Candidate profiling is an optimization. Preserve the known-safe
            # deterministic dispatcher if profiling itself is unavailable.
            _verbose(f"tuning failed; using builtin fallback: {error}")
            result_config = W6A8Config(FALLBACK_CONFIG_ID, 1, 0)

        if result is not None:
            _verbose(
                f"selected {result_config.kernel}, "
                f"{result_config.raster}/sw{result_config.swizzle} "
                f"({result.latency_us:.3f} us; fallback "
                f"{result.fallback_us:.3f} us)"
            )

        _install(a, m, n, k, result_config, out_dtype)
        with _STATE_LOCK:
            _DECISIONS[process_key] = result_config
        return result_config


def cache_directory() -> Path:
    """Return the directory used for persistent native autotune entries."""
    return _cache_root()
