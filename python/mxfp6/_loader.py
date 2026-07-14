from __future__ import annotations

import os
import threading
from pathlib import Path

import torch


_LOAD_LOCK = threading.Lock()
_LOADED_PATH: Path | None = None


def _library_candidates() -> list[Path]:
    candidates: list[Path] = []
    override = os.environ.get("MXFP6_LIBRARY_PATH")
    if override:
        candidates.append(Path(override).expanduser())

    package_dir = Path(__file__).resolve().parent
    candidates.extend(sorted(package_dir.glob("mxfp6_torch*.so")))

    # Source-tree fallback for developers using PYTHONPATH=python.
    source_root = package_dir.parents[1]
    candidates.append(source_root / "build" / "mxfp6_torch.so")
    return candidates


def load_library() -> Path:
    """Load the dispatcher library once and return its resolved path."""
    global _LOADED_PATH
    if _LOADED_PATH is not None:
        return _LOADED_PATH
    with _LOAD_LOCK:
        if _LOADED_PATH is not None:
            return _LOADED_PATH
        candidates = _library_candidates()
        library = next((path for path in candidates if path.is_file()), None)
        if library is None:
            searched = "\n  ".join(str(path) for path in candidates)
            raise ImportError(
                "mxfp6_torch.so was not found. Install the wheel or build the "
                f"CMake target. Searched:\n  {searched}"
            )
        # Importing torch first makes its C++ DSOs available before dlopen of
        # this dispatcher-only extension. The wheel intentionally does not
        # bundle PyTorch or CUDA runtime libraries.
        torch.ops.load_library(str(library))
        _LOADED_PATH = library.resolve()
        return _LOADED_PATH
