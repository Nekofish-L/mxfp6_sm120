"""Freeze native tactic availability, routing arguments and scheduler classes."""
import hashlib
from pathlib import Path
import runpy
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('prune,expected', [
    (False, 'f2d9d6a1eb3d2517d1e22561e4b1f76fe5f9c560a3d1745fd168cd142c5ab3b6'),
    (True, '2df114bf5ff4a68d2a94a5af9b2ede76e7c9c72825284482514523909732da09'),
    ('jit', '9b5349e6730dfeba377553e723651f6afc24f642a865cebaded5e4440a4f4de0'),
])
def test_legacy_tactic_routes(tmp_path, prune, expected):
    # Captured from the independent availability/Stream-K/dispatch branches at
    # 42b7b10, including holes, replicas, GEMV rows and family-local IDs.
    generate = runpy.run_path(str(ROOT/'scripts/generate_mxfp8_dispatch.py'))
    registry = tmp_path/'mxfp8_gemm/registry.def'
    registry.parent.mkdir()
    registry.write_text(generate['generate_registry'](
        ROOT/'python/mxfp6/mxfp8_dispatch.json',
        ROOT/'csrc/include/mxfp8_gemm/registry.def', prune,
        ROOT/'python/mxfp6/mxfp8_cp_async.json' if prune == 'jit' else None))
    source = tmp_path/'routes.cpp'
    source.write_text(r'''
#include <iostream>
#include <limits>
#include "mxfp8_gemm/tactic.hpp"
using namespace mxfp8_runtime;
const char* name(Backend backend) {
  switch (backend) {
#define NAME(N) case Backend::N: return #N
    NAME(invalid); NAME(core); NAME(direct); NAME(dynamic); NAME(extra);
    NAME(wide); NAME(release_core); NAME(release_direct); NAME(gemv);
#undef NAME
#define MXFP8_FAMILY(N, B, C, R) case Backend::N: return #N;
#include "mxfp8_gemm/registry.def"
#undef MXFP8_FAMILY
  }
  return "unknown";
}
void print(int64_t t) {
  const auto decoded = decode_tactic(t);
  std::cout << t << ' ' << name(decoded.backend) << ' '
            << decoded.argument << ' ' << decoded.stream_k << '\n';
}
int main() {
  print(std::numeric_limits<int64_t>::min());
  for (int64_t t = -2; t < 1602; ++t) print(t);
  print(std::numeric_limits<int64_t>::max());
}
''')
    binary = tmp_path/'routes'
    subprocess.run(['c++', '-std=c++17', '-I'+str(tmp_path),
                    '-I'+str(ROOT/'csrc/include'), str(source), '-o', str(binary)], check=True)
    assert hashlib.sha256(subprocess.check_output([str(binary)])).hexdigest() == expected
