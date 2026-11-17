"""Persistent tuning decisions, independent of CUDA and kernel compilation.

The descriptor and on-disk schema remain compatible with existing caches.
Callers own process-local locks; file_lock serializes tuning across processes.
"""
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterator

from ._autotune_policy import (
    FALLBACK_CONFIG_ID, RASTER_NAMES, SWIZZLES, AutotuneResult, W6A8Config,
)
from ._w6a8_registry import KERNEL_NAMES


def cache_key(descriptor: dict[str, object]) -> str:
    canonical = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _entry_path(root: Path, key: str) -> Path:
    return root / f"{key}.json"


@contextmanager
def file_lock(root: Path, key: str) -> Iterator[None]:
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


def read_entry(
    root: Path, key: str, descriptor: dict[str, object]
) -> W6A8Config | None:
    path = _entry_path(root, key)
    try:
        with path.open() as file:
            entry = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(entry, dict) or entry.get("descriptor") != descriptor:
        return None
    return _config_from_dict(entry.get("config"))


def write_entry(
    root: Path, key: str,
    descriptor: dict[str, object],
    result: AutotuneResult,
) -> None:
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
        os.replace(temporary_name, _entry_path(root, key))
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
