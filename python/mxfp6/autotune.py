"""First-use native W6A8 autotuning with a persistent dispatch cache.

The kernels themselves are AOT-compiled into ``mxfp6_torch``.  For a shape
without a checked-in exact override, this module benchmarks a shape-aware
shortlist once, validates the winner, installs it in the C++ dispatcher, and
persists the decision for later processes.  No Humming code is used here.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import tempfile
import threading
import time
from typing import Iterator

import torch

from ._loader import load_library


AUTOTUNE_SCHEMA = 1
CANDIDATE_ABI = "native-w6a8-29-v3"
TIMING_POLICY = "gemm-kernels-two-stage-v3"
FALLBACK_CONFIG_ID = -1

KERNEL_NAMES = (
    "swapped_128x8_static_cooperative",
    "swapped_128x8_stage4_static_cooperative",
    "swapped_128x8_stage5_static_cooperative",
    "swapped_128x8_stream_k",
    "swapped_128x16_stage4_static_cooperative",
    "swapped_64x16x256_static_pingpong",
    "normal_64x64x128_pingpong",
    "normal_64x64x128_static_pingpong",
    "normal_64x64x128_stage4_static_pingpong",
    "normal_64x64x256_static_pingpong",
    "normal_64x128x128_pingpong",
    "normal_64x128x128_static_pingpong",
    "normal_128x128x128_pingpong",
    "normal_128x128x128_cooperative",
    "normal_128x128x128_static_pingpong",
    "normal_128x128x128_static_cooperative",
    "normal_128x128x128_stream_k",
    "swapped_64x32x128_pingpong",
    "swapped_64x32x128_stage3_pingpong",
    "swapped_64x32x128_static_pingpong",
    "swapped_64x32x128_stage3_static_pingpong",
    "swapped_64x32x256_pingpong",
    "swapped_64x32x256_stage2_pingpong",
    "swapped_64x32x256_static_pingpong",
    "swapped_64x32x256_stage2_static_pingpong",
    "swapped_128x32_cooperative",
    "swapped_128x32_stage2_cooperative",
    "swapped_128x32_static_cooperative",
    "swapped_128x32_stage2_static_cooperative",
)
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
    """Return whether first-use tuning is enabled (enabled by default)."""
    return _env_bool("MXFP6_AUTOTUNE", True)


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
        "timing_policy": TIMING_POLICY,
        "library": _library_fingerprint(),
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
            "warmup": _positive_env_int("MXFP6_AUTOTUNE_WARMUP", 2),
            "iterations": _positive_env_int("MXFP6_AUTOTUNE_ITERATIONS", 5),
            "repeats": _positive_env_int("MXFP6_AUTOTUNE_REPEATS", 3),
            "minimum_improvement": float(
                os.environ.get("MXFP6_AUTOTUNE_MIN_IMPROVEMENT", "0.02")
            ),
        },
    }


def _cache_key(descriptor: dict[str, object]) -> str:
    canonical = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _entry_path(key: str) -> Path:
    return _cache_root() / f"{key}.json"


def _get_key_lock(key: str) -> threading.Lock:
    with _STATE_LOCK:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _file_lock(key: str) -> Iterator[None]:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f"{key}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _config_from_dict(value: object) -> W6A8Config | None:
    if not isinstance(value, dict):
        return None
    try:
        config = W6A8Config(
            int(value["config_id"]),
            int(value["swizzle"]),
            int(value["raster_order"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if config.config_id < FALLBACK_CONFIG_ID or config.config_id >= len(
        KERNEL_NAMES
    ):
        return None
    if config.swizzle not in SWIZZLES:
        return None
    if not 0 <= config.raster_order < len(RASTER_NAMES):
        return None
    return config


def _read_entry(
    key: str, descriptor: dict[str, object]
) -> W6A8Config | None:
    path = _entry_path(key)
    try:
        with path.open() as file:
            entry = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(entry, dict) or entry.get("descriptor") != descriptor:
        return None
    return _config_from_dict(entry.get("config"))


def _write_entry(
    key: str,
    descriptor: dict[str, object],
    result: AutotuneResult,
) -> None:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    entry = {
        "descriptor": descriptor,
        "config": {
            **asdict(result.config),
            "kernel": result.config.kernel,
            "raster": result.config.raster,
        },
        "measurement": {
            "latency_us": result.latency_us,
            "fallback_us": result.fallback_us,
            "runner_up_us": result.runner_up_us,
            "samples": result.samples,
        },
        "created_unix": time.time(),
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{key}.", suffix=".tmp", dir=root
    )
    try:
        with os.fdopen(file_descriptor, "w") as file:
            json.dump(entry, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, _entry_path(key))
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


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
    if m <= 16:
        kernels = [0, 1, 2, 3, 4, 5]
    elif m <= 32:
        kernels = [
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        ]
    else:
        kernels = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    if k < 1024:
        kernels = [kernel for kernel in kernels if kernel not in (3, 16)]
    return tuple(kernels)


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
    for _ in range(warmup):
        _run_config(config, a, b, sfa, sfb, m, n, k, out_dtype)
    torch.cuda.synchronize(a.device)
    samples = []
    for _ in range(repeats):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            acc_events=True,
        ) as profiler:
            for _ in range(iterations):
                if flush is not None:
                    flush.zero_()
                _run_config(
                    config, a, b, sfa, sfb, m, n, k, out_dtype
                )
        torch.cuda.synchronize(a.device)
        gemm_device_times = [
            event.self_device_time_total
            for event in profiler.events()
            if event.device_type.name == "CUDA"
            and "cutlass" in event.name.lower()
        ]
        if len(gemm_device_times) < iterations:
            raise RuntimeError(
                f"expected at least {iterations} GEMM device events; "
                f"found {len(gemm_device_times)}"
            )
        samples.append(sum(gemm_device_times) / iterations)
    return statistics.median(samples)


def _candidate_configs(m: int, k: int) -> tuple[list[W6A8Config], list[int]]:
    kernel_ids = _kernel_ids(m, k)
    coarse = [
        W6A8Config(kernel_id, swizzle, 0)
        for kernel_id in kernel_ids
        for swizzle in SWIZZLES
    ]
    return coarse, list(kernel_ids)


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
    warmup = _positive_env_int("MXFP6_AUTOTUNE_WARMUP", 2)
    iterations = _positive_env_int("MXFP6_AUTOTUNE_ITERATIONS", 5)
    repeats = _positive_env_int("MXFP6_AUTOTUNE_REPEATS", 3)
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
    top_kernel_ids = (
        ranked_kernel_ids
        if _use_exhaustive_search()
        else ranked_kernel_ids[:3]
    )
    if not top_kernel_ids:
        top_kernel_ids = kernel_ids[:3]

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

    Returns ``None`` when autotuning is disabled or unsafe in the current
    capture/compile context. A fallback decision is represented by config ID
    ``-1`` and is also cached, preventing repeated unsuccessful searches.
    """
    if not force and not is_autotune_enabled():
        return None
    if _is_compiling_or_capturing():
        return None
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
    if not force:
        with _STATE_LOCK:
            existing = _DECISIONS.get(process_key)
        if existing is not None:
            return existing

    with _get_key_lock(key):
        if not force:
            with _STATE_LOCK:
                existing = _DECISIONS.get(process_key)
            if existing is not None:
                return existing

        cached = None if force else _read_entry(key, descriptor)
        if cached is not None:
            _install(a, m, n, k, cached, out_dtype)
            with _STATE_LOCK:
                _DECISIONS[process_key] = cached
            _verbose(
                f"cache hit {m}x{n}x{k}: {cached.kernel}, "
                f"{cached.raster}/sw{cached.swizzle}"
            )
            return cached

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
