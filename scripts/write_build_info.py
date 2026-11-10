"""Write wheel-local build identities after both CMake libraries exist."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache(build_dir: Path) -> dict[str, str]:
    values = {}
    for line in (build_dir / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(("//", "#")) or "=" not in line or ":" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.split(":", 1)[0]
        if key.startswith(("CMAKE_", "MXFP6_")) or key in ("CUTLASS_DIR", "Torch_DIR", "Python3_EXECUTABLE"):
            values[key] = value
    return values


def _compiler(build_dir: Path, cache: dict[str, str], language: str) -> dict:
    # Compiler identity/version are CMake's detected values, not the shell's
    # default c++ or nvcc, which may differ from the configured toolchain.
    detected = {}
    for path in sorted((build_dir / "CMakeFiles").glob(f"*/CMake{language}Compiler.cmake")):
        for key, value in re.findall(r'^set\((CMAKE_\w+) "([^"\n]*)"\)', path.read_text(), re.MULTILINE):
            if key in (f"CMAKE_{language}_COMPILER", f"CMAKE_{language}_COMPILER_ID", f"CMAKE_{language}_COMPILER_VERSION"):
                detected[key] = value
    executable = detected.get(f"CMAKE_{language}_COMPILER", cache[f"CMAKE_{language}_COMPILER"])
    return {"path": executable, "detected": detected,
            "version_output": subprocess.check_output([executable, "--version"], text=True).strip()}


def write_build_info(output_dir: Path, build_dir: Path, configure: list[str]) -> Path:
    output_dir, build_dir = Path(output_dir), Path(build_dir)
    cache = _cache(build_dir)
    identity = json.loads((build_dir / "torch_build_identity.json").read_text())
    info = {
        "schema_version": 1,
        "runtime": identity["runtime"],
        "python": identity["python"],
        "cmake": {"configure_argv": configure, "cache": cache,
                  "cache_sha256": _sha256(build_dir / "CMakeCache.txt"),
                  "version_output": subprocess.check_output([configure[0], "--version"], text=True).strip()},
        "compilers": {language: _compiler(build_dir, cache, language) for language in ("CXX", "CUDA")},
        "libraries": {name: {"sha256": _sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
                      for name in ("mxfp6_torch.so", "mxfp8_torch.so")},
    }
    path = output_dir / "build_info.json"
    path.write_text(json.dumps(info, indent=2) + "\n")
    return path
