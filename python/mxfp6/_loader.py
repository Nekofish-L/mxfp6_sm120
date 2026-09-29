"""Load the two native dispatcher libraries through the same mechanism."""

from __future__ import annotations

import os
import threading
from pathlib import Path
import torch


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
