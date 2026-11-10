"""Load the two native dispatcher libraries through the same mechanism."""

from __future__ import annotations

import os
import hashlib
import json
import threading
from pathlib import Path
import torch


def _check_packaged_build(library: Path, package: Path) -> None:
    """Reject a wheel compiled against another Torch ABI before loading it."""
    info_path = package / "build_info.json"
    if library.parent.resolve() != package.resolve() or not info_path.is_file():
        return
    info = json.loads(info_path.read_text())
    actual = {
        "torch_version": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    }
    expected = info["runtime"]
    differences = [f"{key}: built={expected[key]!r}, running={value!r}"
                   for key, value in actual.items() if expected[key] != value]
    if differences:
        raise ImportError(
            f"{library.name} has an incompatible packaged build: "
            + "; ".join(differences)
            + ". Install a wheel built for this runtime or rebuild it."
        )
    expected_hash = info["libraries"][library.name]["sha256"]
    if hashlib.sha256(library.read_bytes()).hexdigest() != expected_hash:
        raise ImportError(f"{library.name} does not match its packaged build_info.json")


class _LibraryLoader:
    def __init__(self, name: str):
        self.name = name
        self.path: Path | None = None
        self.lock = threading.Lock()

    def load(self) -> Path:
        if self.path is not None:
            return self.path
        with self.lock:
            if self.path is not None:
                return self.path
            package = Path(__file__).resolve().parent
            root = package.parents[1]
            override = os.environ.get(f"{self.name.upper()}_LIBRARY_PATH")
            candidates = (
                [Path(override).expanduser()]
                if override
                else [
                    *sorted(package.glob(f"{self.name}_torch*.so")),
                    root / "build" / f"{self.name}_torch.so",
                    root / "build" / "mxfp6" / f"{self.name}_torch.so",
                    root / "build" / "mxfp8" / f"{self.name}_torch.so",
                ]
            )
            library = next((p for p in candidates if p.is_file()), None)
            if library is None:
                searched = "\n  ".join(str(p) for p in candidates)
                raise ImportError(
                    f"{self.name}_torch.so was not found. Install the wheel or build the CMake target. Searched:\n  {searched}"
                )
            # Explicit source/manual paths deliberately retain their existing
            # behavior, even if the chosen path happens to be inside package.
            if not override:
                _check_packaged_build(library, package)
            torch.ops.load_library(str(library))
            self.path = library.resolve()
            return self.path


_mxfp6 = _LibraryLoader("mxfp6")
_mxfp8 = _LibraryLoader("mxfp8")


def load_library() -> Path:
    """Load the MXFP6 dispatcher once and return its resolved path."""
    return _mxfp6.load()


def load_mxfp8_library() -> Path:
    """Load the MXFP8 dispatcher once and return its resolved path."""
    return _mxfp8.load()
