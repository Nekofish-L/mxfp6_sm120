"""CPU-only wheel identity and pre-load ABI checks; no Torch/CUDA required."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



class PackageMetadataTests(unittest.TestCase):
    def test_package_metadata_matches_distribution_and_python_version(self):
        project = dict(re.findall(r'^(name|version) = "([^"]+)"$',
                                  (ROOT / "pyproject.toml").read_text(), re.MULTILINE))
        metadata = json.loads((ROOT / "python/mxfp6/metadata.json").read_text())
        module = ast.parse((ROOT / "python/mxfp6/__init__.py").read_text())
        versions = [ast.literal_eval(node.value) for node in module.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "__version__"
                            for target in node.targets)]
        self.assertEqual(metadata["name"], project["name"])
        self.assertEqual(metadata["version"], project["version"])
        self.assertEqual(versions, [project["version"]])


class BuildIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = self.root / "python" / "mxfp6"
        self.package.mkdir(parents=True)
        self.loaded = []
        self.torch = SimpleNamespace(__version__="2.14.0+cu130", version=SimpleNamespace(cuda="13.0"),
                                     _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
                                     ops=SimpleNamespace(load_library=self.loaded.append))
        self.runtime = {"torch_version": self.torch.__version__, "torch_cuda": "13.0", "cxx11_abi": True}
        self.patcher = patch.dict(sys.modules, {"torch": self.torch})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.env = patch.dict(os.environ, {"MXFP6_LIBRARY_PATH": "", "MXFP8_LIBRARY_PATH": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.loader = module_from_file("build_identity_loader", ROOT / "python/mxfp6/_loader.py")
        self.loader.__file__ = str(self.package / "_loader.py")
        self.libs = {}
        for name in ("mxfp6_torch.so", "mxfp8_torch.so"):
            path = self.package / name
            path.write_bytes(name.encode())
            self.libs[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def metadata(self, **runtime):
        info = {"schema_version": 1, "runtime": dict(self.runtime, **runtime), "libraries": self.libs}
        (self.package / "build_info.json").write_text(json.dumps(info))

    def test_matching_wheel_loads_both_libraries_once(self):
        self.metadata()
        for name in ("mxfp6", "mxfp8"):
            loader = self.loader._LibraryLoader(name)
            expected = (self.package / (name + "_torch.so")).resolve()
            self.assertEqual(loader.load(), expected)
            self.assertEqual(loader.load(), expected)
        self.assertEqual(len(self.loaded), 2)

    def test_old_torch_cuda_or_cxx_abi_is_rejected_before_dlopen(self):
        for key, value in [("torch_version", "2.13.0+cu130"), ("torch_cuda", "12.8"), ("cxx11_abi", False)]:
            with self.subTest(key=key):
                self.metadata(**{key: value})
                with self.assertRaisesRegex(ImportError, key):
                    self.loader._LibraryLoader("mxfp6").load()
        self.assertEqual(self.loaded, [])

    def test_wrong_library_cannot_borrow_a_matching_metadata_file(self):
        self.metadata()
        (self.package / "mxfp8_torch.so").write_bytes(b"old library")
        with self.assertRaisesRegex(ImportError, "does not match"):
            self.loader._LibraryLoader("mxfp8").load()
        self.assertEqual(self.loaded, [])

    def test_explicit_source_override_is_not_replaced_or_checked(self):
        self.metadata(torch_version="2.13.0+cu130")
        source = self.root / "manual.so"
        source.write_bytes(b"source build")
        with patch.dict(os.environ, {"MXFP6_LIBRARY_PATH": str(source)}):
            self.assertEqual(self.loader._LibraryLoader("mxfp6").load(), source.resolve())
        # Explicit paths retain the same behavior even when pointing inside package.
        packaged = self.package / "mxfp8_torch.so"
        with patch.dict(os.environ, {"MXFP8_LIBRARY_PATH": str(packaged)}):
            self.assertEqual(self.loader._LibraryLoader("mxfp8").load(), packaged.resolve())
        self.assertEqual(self.loaded, [str(source), str(packaged)])

    def test_manual_build_without_metadata_keeps_existing_behavior(self):
        packaged = self.package / "mxfp6_torch.so"
        self.assertEqual(self.loader._LibraryLoader("mxfp6").load(), packaged.resolve())
        packaged.unlink()
        build = self.root / "build" / "mxfp6_torch.so"
        build.parent.mkdir()
        build.write_bytes(b"manual build")
        self.assertEqual(self.loader._LibraryLoader("mxfp6").load(), build.resolve())

    def test_writer_records_actual_runtime_compilers_and_both_dso_hashes(self):
        writer = module_from_file("build_identity_writer", ROOT / "scripts/write_build_info.py")
        build = self.root / "cmake-build"
        build.mkdir()
        (build / "torch_build_identity.json").write_text(json.dumps(
            {"runtime": self.runtime, "python": {"executable": "/cmake/python"}}))
        (build / "CMakeCache.txt").write_text(
            "CMAKE_CXX_COMPILER:FILEPATH=/toolchain/c++\n"
            "CMAKE_CUDA_COMPILER:FILEPATH=/toolchain/nvcc\n"
            "CMAKE_CUDA_FLAGS:STRING=--use_fast_math\n"
            "CUTLASS_DIR:PATH=/cutlass\n")
        detected = build / "CMakeFiles" / "4.0"
        detected.mkdir(parents=True)
        (detected / "CMakeCUDACompiler.cmake").write_text('set(CMAKE_CUDA_COMPILER_VERSION "13.0.88")\n')
        configure = ["cmake", "-B", str(build), "-DMXFP6_BUILD_STANDALONE=OFF"]
        with patch.object(writer.subprocess, "check_output", side_effect=lambda argv, **kw: "version of " + argv[0]):
            path = writer.write_build_info(self.package, build, configure)
        info = json.loads(path.read_text())
        self.assertEqual(info["runtime"], self.runtime)
        self.assertEqual(info["python"]["executable"], "/cmake/python")
        self.assertEqual(info["cmake"]["configure_argv"], configure)
        self.assertEqual(info["cmake"]["cache"]["CMAKE_CUDA_FLAGS"], "--use_fast_math")
        self.assertEqual(info["compilers"]["CUDA"]["detected"]["CMAKE_CUDA_COMPILER_VERSION"], "13.0.88")
        for name, value in self.libs.items():
            self.assertEqual(info["libraries"][name]["sha256"], value["sha256"])
        self.assertTrue(path.is_relative_to(self.root))
        self.assertFalse((ROOT / "python/mxfp6/build_info.json").exists())

    @unittest.skipUnless(importlib.util.find_spec("setuptools") and importlib.util.find_spec("wheel"),
                         "wheel packaging test requires setuptools and wheel")
    def test_bdist_wheel_installs_generated_metadata_and_both_libraries(self):
        # Exercise the real setuptools build/install/wheel lifecycle while a
        # fake CMake produces fixture DSOs. This tests packaging, not CUDA.
        source = self.root / "source"
        source.mkdir()
        for name in ("setup.py", "pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
            shutil.copyfile(ROOT / name, source / name)
        shutil.copytree(ROOT / "python", source / "python")
        (source / "scripts").mkdir()
        shutil.copyfile(ROOT / "scripts/write_build_info.py", source / "scripts/write_build_info.py")
        tools = self.root / "tools"
        tools.mkdir()
        cmake = tools / "cmake"
        cmake.write_text(f"#!{sys.executable}\n" + '''import json, sys
from pathlib import Path
args = sys.argv[1:]
if args == ['--version']:
    print('fixture compiler/CMake version')
elif '--build' in args:
    build = Path(args[args.index('--build') + 1])
    output = Path((build / 'fixture-output').read_text())
    for name in ('mxfp6_torch.so', 'mxfp8_torch.so'):
        (output / name).write_bytes(('fixture ' + name).encode())
else:
    import torch
    build = Path(args[args.index('-B') + 1])
    output = next(a.split('=', 1)[1] for a in args if a.startswith('-DCMAKE_LIBRARY_OUTPUT_DIRECTORY='))
    (build / 'fixture-output').write_text(output)
    compiler = str(Path(sys.argv[0]).resolve())
    (build / 'CMakeCache.txt').write_text('CMAKE_CXX_COMPILER:FILEPATH=' + compiler + '\\nCMAKE_CUDA_COMPILER:FILEPATH=' + compiler + '\\n')
    (build / 'torch_build_identity.json').write_text(json.dumps({'runtime': {'torch_version': torch.__version__, 'torch_cuda': torch.version.cuda, 'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI}, 'python': {'executable': sys.executable}}))
''')
        cmake.chmod(0o755)
        fake = self.root / "fake-modules"
        fake.mkdir()
        (fake / "torch.py").write_text(
            "from types import SimpleNamespace\n__version__='2.14.0+cu130'\n"
            "version=SimpleNamespace(cuda='13.0')\n_C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)\n")
        env = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""),
                   PYTHONPATH=str(fake), CMAKE_ARGS="", MAX_JOBS="1")
        output = subprocess.run([sys.executable, "setup.py", "bdist_wheel"], cwd=source, env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(output.returncode, 0, output.stdout)
        wheel = next((source / "dist").glob("*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            info = json.loads(archive.read("mxfp6/build_info.json"))
            self.assertEqual(info["runtime"], self.runtime)
            for name in ("mxfp6_torch.so", "mxfp8_torch.so"):
                self.assertEqual(hashlib.sha256(archive.read("mxfp6/" + name)).hexdigest(),
                                 info["libraries"][name]["sha256"])
            metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            self.assertIn("Version: 0.3.0", archive.read(metadata).decode())
        self.assertFalse((source / "python/mxfp6/build_info.json").exists())
        self.assertFalse(list((source / "python/mxfp6").glob("*.so")))


if __name__ == "__main__":
    unittest.main()
