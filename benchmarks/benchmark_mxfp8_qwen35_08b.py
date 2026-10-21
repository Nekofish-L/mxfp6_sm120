#!/usr/bin/env python3
"""Paired SM120 GEMM measurements for Qwen3.5-0.8B TP1 projections.

Uses synthetic operands by default. --checkpoint reads representative compressed
weights without loading the model. Quantization and cache eviction are untimed.
FlashInfer CUTLASS and b12x are independently autotuned on the exact batch size.
"""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

import flashinfer
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from mxfp6 import mxfp8


# Fused runtime projections, with actual layer invocation counts. The two output
# projections share the same N/K; ignored BF16 vision/BA/lm_head are excluded.
PROJECTIONS = [
    ("gdn.out_proj", 1024, 2048, 18),
    ("attn.o_proj", 1024, 2048, 6),
    ("mlp.down_proj", 1024, 3584, 24),
    ("attn.qkv_gate_proj", 5120, 1024, 6),
    ("mlp.gate_up_proj", 7168, 1024, 24),
    ("gdn.in_proj_qkvz", 8192, 1024, 18),
]
BATCHES = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64,
           96, 128, 192, 256, 512, 1024, 2048, 4096, 8192]


def pack_scales(logical):
    """Independent conversion of checkpoint E8M0 bytes to CUTLASS 128x4."""
    rows, groups = logical.shape
    padded = torch.zeros(((rows + 127) // 128 * 128, groups),
                         device=logical.device, dtype=torch.uint8)
    padded[:rows].copy_(logical)
    return padded.reshape(-1, 4, 32, groups // 4, 4).transpose(1, 3).contiguous().flatten()


def checkpoint_weights(path):
    from safetensors import safe_open

    # Concatenation follows the runtime's fused linear projection order.
    groups = {
        (1024, 2048): (0, "linear_attn", ["out_proj"]),
        (1024, 3584): (0, "mlp", ["down_proj"]),
        (5120, 1024): (3, "self_attn", ["q_proj", "k_proj", "v_proj"]),
        (7168, 1024): (0, "mlp", ["gate_proj", "up_proj"]),
        (8192, 1024): (0, "linear_attn", ["in_proj_qkv", "in_proj_z"]),
    }
    result = {}
    with safe_open(str(path / "model.safetensors"), framework="pt") as checkpoint:
        for (n, k), (layer, module, names) in groups.items():
            prefixes = [f"model.language_model.layers.{layer}.{module}.{name}" for name in names]
            values = torch.cat([checkpoint.get_tensor(p + ".weight").view(torch.uint8)
                                for p in prefixes]).cuda()
            scales = torch.cat([checkpoint.get_tensor(p + ".weight_scale").view(torch.uint8)
                                for p in prefixes]).cuda()
            assert values.shape == (n, k) and scales.shape == (n, k // 32)
            result[n, k] = values.view(torch.float8_e4m3fn), pack_scales(scales)
    return result


def policy_config(policy, m, n, k):
    """Interpret a frozen policy for an offline before/after comparison."""
    def resolve(config):
        if isinstance(config, list):
            return mxfp8.W8A8Config(*config)
        value = {"m": m, "n": n, "k": k}[config["axis"]]
        return resolve(config["le" if value <= config["upper"] else "gt"])

    for region in policy["regions"]:
        if all((region[axis + "_max"] is None or value <= region[axis + "_max"])
               and value >= region.get(axis + "_min", 1)
               for axis, value in (("n", n), ("k", k))):
            for bound, config in zip(policy["m_bounds"], region["configs"]):
                if m <= bound:
                    return resolve(config)
            return mxfp8.W8A8Config(79)
    raise ValueError("Shape outside the frozen policy regions")


def capture(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    graph._prepared_callable = fn
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def profile_trial(graphs, eviction, tests, rng):
    """Time exactly one kernel per replay, with randomized backend order."""
    names = list(graphs)
    order = []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(tests):
            rng.shuffle(names)
            for name in names:
                eviction.zero_()
                graphs[name][0].replay()
                order.append(name)
        torch.cuda.synchronize()
    events = sorted((event for event in prof.events()
                     if event.device_type == torch.autograd.DeviceType.CUDA
                     and "memset" not in event.name.lower()
                     and "FillFunctor" not in event.name),
                    key=lambda event: event.time_range.start)
    # This also rejects a baseline that unexpectedly launches extra kernels.
    if len(events) != len(order):
        raise RuntimeError(f"Expected {len(order)} GEMM kernels, got {len(events)}: "
                           f"{[event.name for event in events[:8]]}")
    samples = {name: [] for name in names}
    kernels = {}
    for name, event in zip(order, events):
        samples[name].append(event.device_time_total)
        kernels[name] = event.name
    return samples, kernels


def relative_rmse(output, reference):
    assert torch.isfinite(output).all()
    return ((output.float() - reference.float()).square().sum()
            / reference.float().square().sum()).sqrt().item()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    counts = {}
    for _, n, k, count in PROJECTIONS:
        counts[n, k] = counts.get((n, k), 0) + count
    comparisons = {}
    for baseline in ("before", "cutlass", "b12x", "fastest_flashinfer"):
        if baseline == "before" and "before" not in rows[0]["backends"]:
            continue
        ratios = []
        slower = []
        totals = {}
        for row in rows:
            backends = row["backends"]
            native = backends["native"]["us"]
            reference = (min(backends[name]["us"] for name in ("cutlass", "b12x"))
                         if baseline == "fastest_flashinfer" else backends[baseline]["us"])
            ratios.append(reference / native)
            if native >= reference:
                slower.append(dict(m=row["m"], n=row["n"], k=row["k"],
                                   slowdown=native / reference))
            total = totals.setdefault(row["m"], dict(native_us=0., reference_us=0.))
            count = counts[row["n"], row["k"]]
            total["native_us"] += count * native
            total["reference_us"] += count * reference
        for total in totals.values():
            total["speedup"] = total["reference_us"] / total["native_us"]
        comparisons[baseline] = dict(
            geometric_mean_speedup=math.exp(statistics.mean(map(math.log, ratios))),
            faster_points=len(rows) - len(slower), points=len(rows),
            slower_points=slower, model_gemm_totals_by_batch=totals)
    return comparisons


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    parser.add_argument("--before-policy", type=Path)
    parser.add_argument("--candidates", type=Path,
                        help='Optional JSON map from "M,N,K" to lists of launch configurations')
    parser.add_argument("--tests", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--eviction-mib", type=int, default=256)
    parser.add_argument("--check-graphs", action="store_true")
    args = parser.parse_args()
    if min(args.tests, args.trials, *args.batches) <= 0:
        parser.error("tests, trials and batches must be positive")
    properties = torch.cuda.get_device_properties(0)
    if args.eviction_mib * 1024 ** 2 <= properties.L2_cache_size:
        parser.error("eviction buffer must exceed the GPU L2 cache")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261021)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = random.Random(20261021)
    eviction = torch.empty(args.eviction_mib * 1024 ** 2, device="cuda", dtype=torch.uint8)
    before = json.loads(args.before_policy.read_text()) if args.before_policy else None
    candidates = json.loads(args.candidates.read_text()) if args.candidates else {}
    weights = checkpoint_weights(args.checkpoint) if args.checkpoint else {}
    library = mxfp8.load_library()
    sources = [Path(__file__), ROOT / "python/mxfp6/mxfp8_dispatch.json",
               ROOT / "scripts/generate_mxfp8_dispatch.py",
               ROOT / "csrc/include/mxfp8_gemm/small_batch.cuh",
               ROOT / "csrc/include/mxfp8_gemm/registry.def"]
    metadata = dict(
        gpu=str(properties), visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
        torch=torch.__version__, cuda=torch.version.cuda, flashinfer=flashinfer.__version__,
        native_library_sha256=sha256(library),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sources},
        checkpoint_config_sha256=sha256(args.checkpoint / "config.json") if args.checkpoint else None,
        checkpoint_operand_sha256={
            f"{n},{k}": [hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                         for t in operands]
            for (n, k), operands in weights.items()},
        before_policy_sha256=sha256(args.before_policy) if args.before_policy else None,
        checkpoint_weights=bool(args.checkpoint), batches=args.batches,
        trials=args.trials, tests=args.tests, eviction_mib=args.eviction_mib,
        method="Single-GEMM CUDA graph; randomized paired cold-L2 kernel times; "
               "median of trial means; quantization/eviction/host overhead excluded",
        projections=PROJECTIONS,
    )
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.out / "raw.jsonl").open("w", buffering=1) as log:
        for n, k in sorted({(p[1], p[2]) for p in PROJECTIONS}):
            if (n, k) in weights:
                b, sb = weights[n, k]
            else:
                b, sb = flashinfer.mxfp8_quantize(
                    torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
                    is_sf_swizzled_layout=True)
            qb = mxfp8.MXFP8Tensor(b.view(torch.uint8), sb, n, k)
            for m in args.batches:
                a, sa = flashinfer.mxfp8_quantize(
                    torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                    is_sf_swizzled_layout=True)
                qa = mxfp8.MXFP8Tensor(a.view(torch.uint8), sa, m, k)
                native = mxfp8.prepare(qa, qb)
                fns = {"native": native}
                configs = {"native": list(native.config)}
                if before:
                    config = policy_config(before, m, n, k)
                    fns["before"] = mxfp8.prepare(qa, qb, config=config)
                    configs["before"] = list(config)
                for config in candidates.get(f"{m},{n},{k}", []):
                    name = ",".join(map(str, config))
                    fns[name] = mxfp8.prepare(qa, qb, config=mxfp8.W8A8Config(*config))
                    configs[name] = config
                for backend in ("cutlass", "b12x"):
                    fns[backend] = lambda backend=backend: flashinfer.mm_mxfp8(
                        a, b.T, sa, sb, out_dtype=torch.bfloat16,
                        backend=backend, use_8x4_sf_layout=False)
                    with flashinfer.autotune(True, tuning_buckets=(m,)):
                        fns[backend]()
                with flashinfer.autotune(False, tuning_buckets=(m,)):
                    graphs = {}
                    for name, fn in fns.items():
                        try:
                            graphs[name] = capture(fn)
                        except Exception as error:
                            raise RuntimeError(f"Capture failed: M={m}, N={n}, K={k}, "
                                               f"backend={name}") from error
                    reference = graphs["cutlass"][1].clone()
                    errors = {name: relative_rmse(out, reference)
                              for name, (_, out) in graphs.items()}
                    assert max(errors.values()) < .004, (m, n, k, errors)
                    samples = {name: [] for name in graphs}
                    means = {name: [] for name in graphs}
                    for _ in range(args.trials):
                        trial, kernels = profile_trial(graphs, eviction, args.tests, rng)
                        for name, values in trial.items():
                            samples[name].extend(values)
                            means[name].append(statistics.mean(values))
                    changing_error = None
                    if args.check_graphs:
                        changing_error = 0.
                        for exponent in (-5, 4):
                            na, nsa = flashinfer.mxfp8_quantize(
                                torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 2. ** exponent,
                                is_sf_swizzled_layout=True)
                            a.copy_(na)
                            sa.copy_(nsa)
                            expected = fns["cutlass"]().clone()
                            for graph, out in graphs.values():
                                graph.replay()
                                error = relative_rmse(out, expected)
                                assert error < .004, (m, n, k, error)
                                changing_error = max(changing_error, error)
                        a.zero_()
                        for graph, out in graphs.values():
                            graph.replay()
                            assert torch.count_nonzero(out).item() == 0
                    backends = {name: dict(us=statistics.median(means[name]),
                                           trial_means_us=means[name], samples_us=samples[name],
                                           kernel=kernels[name], relative_rmse=errors[name])
                                for name in graphs}
                    row = dict(m=m, n=n, k=k, configs=configs, backends=backends,
                               changing_graph_rmse=changing_error)
                    log.write(json.dumps(row) + "\n")
                    print(m, n, k, {name: round(value["us"], 3)
                                   for name, value in backends.items()}, flush=True)
    rows = [json.loads(line) for line in (args.out / "raw.jsonl").read_text().splitlines()]
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
