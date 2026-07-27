from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        output_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build_dir = Path(self.build_temp).resolve() / ext.name.replace(".", "_")
        output_dir.mkdir(parents=True, exist_ok=True)
        build_dir.mkdir(parents=True, exist_ok=True)

        configure = [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DMXFP6_BUILD_STANDALONE=OFF",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if "CMAKE_GENERATOR" not in os.environ and shutil.which("ninja"):
            configure.extend(["-G", "Ninja"])
        configure.extend(shlex.split(os.environ.get("CMAKE_ARGS", "")))
        subprocess.check_call(configure, cwd=ROOT)

        jobs = os.environ.get("MAX_JOBS", "2")
        subprocess.check_call(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--target",
                "mxfp6_torch",
                "--parallel",
                jobs,
            ],
            cwd=ROOT,
        )


setup(
    ext_modules=[CMakeExtension("mxfp6.mxfp6_torch")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
