"""Public API and native workspace regressions shared with the MXFP6 workflow."""

import hashlib
import json
from pathlib import Path
import pytest
import torch
import mxfp6.mxfp8 as mxfp8

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,n,k", [(3, 128, 128), (17, 256, 512), (65, 384, 1024)])
def test_public_gemm_and_float_activation(m, n, k, dtype):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    qa = mxfp8.quantize_activation(a)
    qb = mxfp8.quantize_mxfp8(b)
    expected = mxfp8.gemm(qa, qb)
    torch.testing.assert_close(mxfp8.gemm(a, qb), expected, rtol=0, atol=0)
    torch.testing.assert_close(mxfp8.gemm_from_float(a, qb), expected, rtol=0, atol=0)
    torch.testing.assert_close(mxfp8.gemm_w8a8(qa, qb), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        mxfp8.gemm_packed(qa.values, qb.values, qa.scales, qb.scales, m, n, k),
        expected,
        rtol=0,
        atol=0,
    )
    assert mxfp8.warmup(a, qb, iterations=1) == mxfp8.select_config(m, n, k)
    for alpha, dtype in [(2.0, torch.bfloat16), (1.0, torch.float16)]:
        with pytest.raises(ValueError):
            mxfp8.gemm(qa, qb, alpha, out_dtype=dtype)


def test_compiled_policy_matches_source_intervals():
    mxfp8.load_library()
    path = ROOT / "python/mxfp6/mxfp8_dispatch.json"
    policy = json.loads(path.read_text())
    assert (
        torch.ops.mxfp8_sm120.policy_sha256()
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    # Include both sides of every threshold and unseen combinations of axes.
    dimensions = {"m": {1, 2049, 8192}, "n": {128, 32768}, "k": {128, 12288}}
    for v in policy["m_bounds"]:
        dimensions["m"].update([v, v + 1])

    def visit(c):
        if isinstance(c, dict):
            dimensions[c["axis"]].update([c["upper"], c["upper"] + 1])
            visit(c["le"])
            visit(c["gt"])

    for r in policy["regions"]:
        for axis in ("n", "k"):
            if axis + "_min" in r:
                dimensions[axis].update([r[axis + "_min"] - 1, r[axis + "_min"]])
            if r[axis + "_max"] is not None:
                dimensions[axis].update([r[axis + "_max"], r[axis + "_max"] + 1])
        for c in r["configs"]:
            visit(c)

    def expected(m, n, k):
        for r in policy["regions"]:
            if (r["n_max"] is None or n <= r["n_max"]) and (
                r["k_max"] is None or k <= r["k_max"]
            ) and n >= r.get("n_min", 1) and k >= r.get("k_min", 1):
                for bound, c in zip(policy["m_bounds"], r["configs"]):
                    if m <= bound:
                        while isinstance(c, dict):
                            c = c[
                                (
                                    "le"
                                    if dict(m=m, n=n, k=k)[c["axis"]] <= c["upper"]
                                    else "gt"
                                )
                            ]
                        return tuple(c)
                return tuple(policy["default_config"])
        return tuple(policy["default_config"])

    for m in dimensions["m"]:
        for n in dimensions["n"]:
            for k in dimensions["k"]:
                assert tuple(mxfp8.select_config(m, n, k)) == expected(m, n, k)
