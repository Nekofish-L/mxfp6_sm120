"""Single/dual timings inside complete Qwen3.5 inference, with natural caches.

Use an isolated, validated vllm-mach champion installation. The only arithmetic
change is the 32 gate/up + 24 QKVZ activation representation. Timing events are
external CUDA Graph nodes, so their timestamps update on real graph replay.
No weight replay loop, cache flush, random activation, or shadow GEMM is used.
"""
from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import time
import torch

_EVENTS = []
_ANCHOR = None
_PATCHED = False
_ACTIVE = False
_FORWARD_EVENTS = None
_FORWARD_M = None


def install_measurement():
    """Called by the isolated profile hook before custom-op registration."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    from vllm_mach.mxfp8 import dual

    def projection(kind, x, weight, scale, layer_id):
        m = int(x.shape[0])
        capture = torch.cuda.is_current_stream_capturing()
        # Preserve the champion's capture coverage checks for both arms.
        if kind == "gateup":
            dual._MLP_COUNTS[("normal_capture" if capture else "normal_eager", m)] += 1
            if capture and m == 32:
                dual._MLP_CAPTURE_LAYERS[layer_id] += 1
        else:
            dual._COUNTS[("capture" if capture else "eager", m)] += 1
            if capture and m in (32, 64):
                dual._CAPTURE_LAYERS[m][layer_id] += 1
        if m > 128:
            return dual._quant_native(x, weight, scale)
        measure = os.environ.get("MX8_INFERENCE_EVENTS", "1") == "1" and (capture or _ACTIVE)
        events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)] if measure else None
        if events:
            events[0].record()
        if os.environ["MX8_INFERENCE_ARM"] == "dual":
            hi, s_hi, residual, s_res = dual._API.quantize(x)
            out = torch.empty((m, weight.shape[0]), device=x.device, dtype=torch.bfloat16)
            if events:
                events[1].record()
            dual._API.gemm_out(hi, weight, s_hi, scale.view(-1), residual, s_res, out)
        else:
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
            a, sa = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            if events:
                events[1].record()
            out = dual.native.gemm(a, weight, sa, scale)
        if events:
            events[2].record()
            _EVENTS.append((kind, layer_id, m, events))
        return out

    def gateup(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, layer_id: int) -> torch.Tensor:
        return projection("gateup", x, weight, scale, layer_id)

    def qkvz(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, layer_id: int) -> torch.Tensor:
        return projection("qkvz", x, weight, scale, layer_id)

    dual._gateup_impl = gateup
    dual._normal_impl = qkvz


def wrap_worker():
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker, "_dual_inference_measurement", False):
        return
    Worker._dual_inference_measurement = True
    original_compile = Worker.compile_or_warm_up_model
    original_execute = Worker.execute_model
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    original_replay = ModelCudaGraphManager.run_fullgraph

    @wraps(original_replay)
    def replay(self, *args, **kwargs):
        global _FORWARD_EVENTS, _FORWARD_M
        if not _ACTIVE:
            return original_replay(self, *args, **kwargs)
        start, stop = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        output = original_replay(self, *args, **kwargs)
        stop.record()
        _FORWARD_EVENTS = (start, stop)
        _FORWARD_M = args[0].num_tokens
        return output

    ModelCudaGraphManager.run_fullgraph = replay

    @wraps(original_compile)
    def compile_model(self, *args, **kwargs):
        global _ANCHOR
        _ANCHOR = torch.cuda.Event(enable_timing=True)
        _ANCHOR.record()
        _ANCHOR.synchronize()
        result = original_compile(self, *args, **kwargs)
        self._dual_inference_iteration = 0
        return result

    @wraps(original_execute)
    def execute(self, scheduler_output, *args, **kwargs):
        global _ACTIVE, _FORWARD_EVENTS, _FORWARD_M
        if not getattr(self, "_mach_mxfp8_ready", False):
            return original_execute(self, scheduler_output, *args, **kwargs)
        directory = Path(os.environ["MX8_INFERENCE_DIR"])
        control = json.loads((directory / "control.json").read_text())
        counts = scheduler_output.num_scheduled_tokens
        pure_decode = bool(counts) and all(v == 1 for v in counts.values())
        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        begin.record()
        _FORWARD_EVENTS = None
        _FORWARD_M = None
        _ACTIVE = True
        try:
            result = original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            _ACTIVE = False
        end.record()
        end.synchronize()
        rows = []
        step_begin = _ANCHOR.elapsed_time(begin)
        step_end = _ANCHOR.elapsed_time(end)
        physical_m = None
        if pure_decode:
            from vllm_mach.mxfp8.graph_policy import CAPTURE_SIZES
            physical_m = next(size for size in CAPTURE_SIZES if size >= sum(counts.values()))
        for kind, layer, m, ev in _EVENTS:
            if physical_m is not None and m != physical_m:
                continue
            if not ev[-1].query():
                continue
            try:
                stamp = _ANCHOR.elapsed_time(ev[-1])
            except torch.AcceleratorError as error:
                # An external event node has no timestamp before its first
                # replay; query() can still return True for that unused event.
                if "invalid resource handle" not in str(error):
                    raise
                continue
            if not step_begin <= stamp <= step_end:
                continue
            rows.append({"kind": kind, "layer": layer, "m": m,
                         "quant_us": ev[0].elapsed_time(ev[1]) * 1000,
                         "gemm_us": ev[1].elapsed_time(ev[2]) * 1000,
                         "total_us": ev[0].elapsed_time(ev[2]) * 1000})
        self._dual_inference_iteration += 1
        record = {**control, "iteration": self._dual_inference_iteration,
                  "requests": len(counts), "tokens": sum(counts.values()),
                  "pure_decode": pure_decode, "forward_ms": begin.elapsed_time(end),
                  "backbone_ms": (_FORWARD_EVENTS[0].elapsed_time(_FORWARD_EVENTS[1])
                                  if _FORWARD_EVENTS else None),
                  "physical_m": _FORWARD_M,
                  "projections": rows}
        with (directory / f"steps-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        return result

    Worker.compile_or_warm_up_model = compile_model
    Worker.execute_model = execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("single", "dual"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Local MXFP8 model directory")
    parser.add_argument("--prompts", type=Path, required=True, help="Local tokenized prompt JSON")
    parser.add_argument("--batches", type=int, nargs="+", default=list(range(1,129)))
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--extra-context", type=int)
    parser.add_argument("--extra-batches", type=int, nargs="+", default=[1,8,16,32,64,96,128])
    parser.add_argument("--extra-output-tokens", type=int, default=384)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(MX8_INFERENCE_ARM=args.arm, MX8_INFERENCE_DIR=str(args.output.resolve()))
    from vllm_mach.mxfp8.serve import configure_environment
    configure_environment(args.output)
    from vllm_mach.mxfp8.profile import check_environment, check_runtime_sources
    versions, sources = check_environment(), check_runtime_sources()
    from vllm_mach.mxfp8.graph_policy import compilation_settings
    settings = dict(model=args.model, tokenizer=args.model, dtype="bfloat16",
                    tensor_parallel_size=1, language_model_only=True,
                    max_model_len=8192, max_num_seqs=128, max_num_batched_tokens=2048,
                    gpu_memory_utilization=0.85, enable_prefix_caching=False,
                    mamba_ssm_cache_dtype="float32", attention_backend="FLASHINFER",
                    quantization="compressed-tensors", kernel_config={"linear_backend":"flashinfer_cutlass"},
                    kv_cache_memory_bytes=19*2**30, kv_cache_dtype="fp8_e4m3",
                    seed=20260907, compilation_config=compilation_settings(), disable_log_stats=True)
    (args.output / "control.json").write_text(json.dumps({"batch":0,"repeat":-1,"warmup":True}))
    (args.output / "run.json").write_text(json.dumps({"arm":args.arm,"versions":versions,"runtime_sources":sources,
                    "settings":settings,"context":args.context,"output_tokens":args.output_tokens,
                    "batches":args.batches,"repeats":args.repeats,
                    "extra_context":args.extra_context,"extra_batches":args.extra_batches,
                    "extra_output_tokens":args.extra_output_tokens,
                    "projection_events":os.environ.get("MX8_INFERENCE_EVENTS","1")=="1",
                    "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "native_sha256":hashlib.sha256(Path(os.environ["MXFP8_LIBRARY_PATH"]).read_bytes()).hexdigest(),
                    "prompt_sha256":hashlib.sha256(args.prompts.read_bytes()).hexdigest()},indent=2))
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(**settings)
    source = json.loads(args.prompts.read_text())["prompts"]
    # Truncated genuine conversations; never repeat short prompts to fill length.
    def prompts_for_context(context):
        candidates = [p for p in source if len(p["token_ids"]) >= context]
        if not candidates:
            raise ValueError(f"No genuine prompt has {context} tokens")
        return [TokensPrompt(prompt_token_ids=candidates[i % len(candidates)]["token_ids"][:context]) for i in range(128)]

    prompts = prompts_for_context(args.context)
    # Warm real inference through representative full-decode graph buckets.
    for batch in (1, 4, 8, 32, 64, 128):
        llm.generate(prompts[:batch], SamplingParams(temperature=0,max_tokens=4,min_tokens=4,ignore_eos=True),use_tqdm=False)
    plans = [(args.context,args.batches,args.output_tokens)]
    if args.extra_context:
        plans.append((args.extra_context,args.extra_batches,args.extra_output_tokens))
    for context, batches, output_tokens in plans:
        prompts = prompts_for_context(context)
        params = SamplingParams(temperature=0,max_tokens=output_tokens,min_tokens=output_tokens,ignore_eos=True,seed=20260907)
        for repeat in range(args.repeats):
            # Reverse the second sweep to reduce within-run drift bias.
            for batch in (batches if repeat % 2 == 0 else list(reversed(batches))):
                control = {"batch":batch,"repeat":repeat,"warmup":False,"context":context}
                temporary=args.output/"control.tmp"
                temporary.write_text(json.dumps(control));temporary.replace(args.output/"control.json")
                started=time.perf_counter()
                results=llm.generate(prompts[:batch],params,use_tqdm=False)
                assert len(results)==batch and all(len(r.outputs[0].token_ids)==output_tokens for r in results)
                response={**control,"wall_s":time.perf_counter()-started,"outputs":[r.outputs[0].token_ids for r in results]}
                with (args.output/"requests.jsonl").open("a") as stream:stream.write(json.dumps(response)+"\n")
                print(f"COMPLETE {args.arm} context={context} repeat={repeat} BS={batch} {response['wall_s']:.3f}s",flush=True)


if __name__ == "__main__":
    main()
