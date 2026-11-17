"""CPU-only checks for the behavior-preserving W6A8 registry migration."""
import ast
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generate_w6a8_dispatch", ROOT / "scripts/generate_w6a8_dispatch.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_legacy_dispatch_tokens():
    # Frozen from the pre-refactor switch and exact-policy bodies. Token-level
    # equality preserves every branch, direct launch, argument and fallback;
    # checking a finite grid of shapes alone would miss expression changes.
    expected = {
        "w6a8_candidates.inc": "ad7c842ec353f383960f47c748627235e9fe49c04feb78844ed6ca70308b21ce",
        "w6a8_exact.inc": "07a7e35f558501835a6ab59edc7263b39a62931f87e367b60840429c71942e45",
    }
    policy = generator.load_policy(generator.DEFAULT_SOURCE)
    generated = generator.generate(policy)
    # Reconstruct the old launch spelling from the selected config. This keeps
    # the original frozen oracle while allowing selection/execution separation.
    generated["w6a8_exact.inc"] = re.sub(
        r"return Config\{(\d+), (.*?), static_cast<int64_t>\((.*?)\)\};",
        lambda m: "return " + generator.launch_call(
            policy["candidates"][int(m[1])], m[2], m[3]) + ";",
        generated["w6a8_exact.inc"],
    ).replace("half_output", "current_output_dtype == at::kHalf")
    for name, digest in expected.items():
        source = re.sub(r"//[^\n]*", "", generated[name])
        tokens = re.findall(r"\w+|[^\s]", source)
        assert hashlib.sha256(" ".join(tokens).encode()).hexdigest() == digest


def test_python_registry_is_current():
    generated = generator.generate(generator.load_policy(generator.DEFAULT_SOURCE))
    assert (ROOT / "python/mxfp6/_w6a8_registry.py").read_text() == generated["_w6a8_registry.py"]
    namespace = {}
    exec(generated["_w6a8_registry.py"], namespace)
    assert len(namespace["KERNEL_NAMES"]) == 30
    assert namespace["CANDIDATE_ABI"] == "native-w6a8-30-v5"
    assert namespace["STREAM_K_CONFIG_IDS"] == {3, 16}


def test_manifest_matches_kernel_traits():
    policy = generator.load_policy(generator.DEFAULT_SOURCE)
    for candidate in policy["candidates"]:
        launch = candidate["launch"]
        orientation = "normal" if launch["kind"] == "normal_128" else launch["kind"]
        alias = ("KernelW6A8_128x128x128" + launch["kernel"]
                 if launch["kind"] == "normal_128" else launch["kernel"])
        header = (ROOT / f"csrc/include/mxfp6_gemm/kernel_{orientation}.hpp").read_text()
        definition = re.search(r"using " + alias + r" = KernelConfig<(.*?);", header, re.S)[1]
        assert [int(v) for v in re.findall(r"cute::_(\d+)", definition)[:3]] == candidate["tile"]
        stages = re.search(r"StageCount<(\d+)>", definition)
        assert (int(stages[1]) if stages else "auto") == candidate["stages"]
        scheduler = ("stream_k" if "StreamKScheduler" in definition else
                     "static" if "PersistentScheduler" in definition else "default")
        assert scheduler == candidate["scheduler"]


@pytest.mark.parametrize("mutation", ["ids", "names", "fallback", "rule", "launch", "tile"])
def test_reject_invalid_manifest(tmp_path, mutation):
    policy = copy.deepcopy(generator.load_policy(generator.DEFAULT_SOURCE))
    if mutation == "ids":
        policy["candidates"][0]["id"] = 1
    elif mutation == "names":
        policy["candidates"][0]["name"] = policy["candidates"][1]["name"]
    elif mutation == "fallback":
        policy["candidates"][3]["disabled_stream_k_fallback"] = 999
    elif mutation == "rule":
        policy["exact_rules"].append({"config": 999, "swizzle": "1", "raster": "RasterOrder::AlongM"})
    elif mutation == "launch":
        policy["candidates"][0]["launch"]["kind"] = "unknown"
    else:
        policy["candidates"][0]["tile"] = [0, 128, 128]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        generator.load_policy(path)


def test_builtin_policy_grid(tmp_path):
    # Frozen from the old direct-launch policy with launch stubs, before the
    # selector extraction. Covers every M through 8193, both dtypes/Stream-K
    # modes, seven SM counts, target NK pairs and non-target fallback shapes.
    generated = generator.generate(generator.load_policy(generator.DEFAULT_SOURCE))
    include = tmp_path / "mxfp6_gemm"
    include.mkdir()
    (include / "w6a8_exact.inc").write_text(generated["w6a8_exact.inc"])
    source = tmp_path / "policy.cpp"
    source.write_text(r'''#include <cstdio>
#include <initializer_list>
#include "mxfp6_gemm/dispatch.hpp"
int main() {
  for (int sm : {1, 2, 16, 80, 128, 170, 192})
  for (bool half : {false, true}) for (bool sk : {false, true})
  for (int n : {8, 136, 256, 1536, 5120, 7168, 8192, 17408})
  for (int k : {128, 384, 1920, 2048, 3072, 5120, 8704})
  for (int m = 1; m <= 8193; ++m) {
    auto c = mxfp6_dispatch::select_builtin(m, n, k, sm, half, sk);
    printf("%lld,%lld,%lld\n", (long long)c.config_id,
           (long long)c.swizzle, (long long)c.raster_order);
  }
}
''')
    binary = tmp_path / "policy"
    subprocess.run(["c++", "-O2", "-std=c++17", "-I" + str(tmp_path),
                    "-I" + str(ROOT / "csrc/include"), str(source),
                    "-o", str(binary)], check=True)
    process = subprocess.Popen([str(binary)], stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    while chunk := process.stdout.read(1024 * 1024):
        digest.update(chunk)
    assert process.wait() == 0
    assert digest.hexdigest() == "f8f05a6ff5917ce9c4fd58e90826e1763da03989a00a49df517efc40f656f85f"
