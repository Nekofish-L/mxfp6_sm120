#!/usr/bin/env python3
"""Compare the SM120 MXFP6 MoE path with vLLM's FP8-block path."""

from __future__ import annotations

import argparse
import statistics

import torch

import mxfp6


def _make_moe_config(
    num_experts: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    max_num_tokens: int,
    dtype: torch.dtype,
):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=dtype,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=max_num_tokens,
    )


def _quantize_mxfp6_weight(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts, rows, k = shape
    source = torch.randn(shape, device="cuda", dtype=dtype) * 0.02
    operand = mxfp6.quantize_mxfp6(source.flatten(0, 1).contiguous())
    if k == 64:
        values = mxfp6.to_mma_k64_weight(operand.values.view(experts, rows, k * 3 // 4))
    else:
        values = operand.values.view(experts, rows, k * 3 // 4)
    scales = operand.scales.view(experts, -1)
    return values, scales


def _make_weights(args, dtype: torch.dtype):
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.experts.mxfp6_sm120_moe import (
        make_mxfp6_sm120_moe_kernel,
    )

    w1_shape = (args.experts, 2 * args.intermediate_size, args.hidden_size)
    w2_shape = (args.experts, args.hidden_size, args.intermediate_size)
    w1_mx, w1_mx_scale = _quantize_mxfp6_weight(w1_shape, dtype)
    w2_mx, w2_mx_scale = _quantize_mxfp6_weight(w2_shape, dtype)
    mx_quant = FusedMoEQuantConfig.make(
        quant_dtype=None,
        weight_dtype="mxfp6_e3m2",
        w1_scale=w1_mx_scale,
        w2_scale=w2_mx_scale,
    )
    moe_config = _make_moe_config(
        args.experts,
        args.topk,
        args.hidden_size,
        args.intermediate_size,
        max(args.batch_sizes),
        dtype,
    )
    mx_kernel = make_mxfp6_sm120_moe_kernel(
        mx_quant,
        moe_config,
        routing_tables=None,
    )

    fp8_intermediate_size = args.fp8_intermediate_size
    fp8_w1_shape = (
        args.experts,
        2 * fp8_intermediate_size,
        args.hidden_size,
    )
    fp8_w2_shape = (
        args.experts,
        args.hidden_size,
        fp8_intermediate_size,
    )
    w1_fp8 = (torch.randn(fp8_w1_shape, device="cuda", dtype=dtype) * 0.02).to(
        torch.float8_e4m3fn
    )
    w2_fp8 = (torch.randn(fp8_w2_shape, device="cuda", dtype=dtype) * 0.02).to(
        torch.float8_e4m3fn
    )
    block = 128
    w1_fp8_scale = torch.full(
        (
            args.experts,
            (2 * fp8_intermediate_size + block - 1) // block,
            (args.hidden_size + block - 1) // block,
        ),
        0.01,
        device="cuda",
        dtype=torch.float32,
    )
    w2_fp8_scale = torch.full(
        (
            args.experts,
            (args.hidden_size + block - 1) // block,
            (fp8_intermediate_size + block - 1) // block,
        ),
        0.01,
        device="cuda",
        dtype=torch.float32,
    )
    fp8_quant = FusedMoEQuantConfig.make(
        quant_dtype=torch.float8_e4m3fn,
        w1_scale=w1_fp8_scale,
        w2_scale=w2_fp8_scale,
        block_shape=[block, block],
    )
    return (
        moe_config,
        mx_kernel,
        (w1_mx, w2_mx),
        (w1_fp8, w2_fp8),
        fp8_quant,
    )


def _bench(
    fn,
    warmup: int,
    iterations: int,
    repeats: int,
    graph_calls: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = None
    if graph_calls > 0:
        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(graph_calls):
                fn()
        torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            if graph is None:
                fn()
            else:
                graph.replay()
        end.record()
        end.synchronize()
        calls = iterations * max(graph_calls, 1)
        samples.append(start.elapsed_time(end) / calls)
    return statistics.median(samples)


def _profile_mxfp6(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor],
    scales: tuple[torch.Tensor, torch.Tensor],
    args,
) -> None:
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
        moe_permute,
        moe_unpermute,
    )

    quantized, logical_scales = mxfp6.quantize_mxfp8_logical(hidden_states)
    (
        permuted,
        permuted_scales,
        expert_offsets,
        inverse_permutation,
        _,
    ) = moe_permute(
        quantized,
        logical_scales,
        topk_ids,
        args.experts,
        args.experts,
    )
    assert permuted_scales is not None
    packed_scales, scale_offsets = mxfp6.pack_grouped_scales(
        permuted_scales, expert_offsets
    )
    gemm1_output = torch.empty(
        (permuted.shape[0], weights[0].shape[1]),
        device="cuda",
        dtype=hidden_states.dtype,
    )
    mxfp6.grouped_gemm_w6a8_out(
        gemm1_output,
        permuted,
        packed_scales,
        weights[0],
        scales[0],
        expert_offsets,
        scale_offsets,
    )
    activation, activation_scales, activation_scale_offsets = (
        mxfp6.silu_and_mul_mxfp8_grouped(gemm1_output, expert_offsets)
    )
    gemm2_output = torch.empty_like(permuted, dtype=hidden_states.dtype)
    array_gemm1_output = torch.empty_like(gemm1_output)
    mxfp6.array_gemm_w6a8_out(
        array_gemm1_output,
        quantized,
        logical_scales,
        weights[0],
        scales[0],
        topk_ids,
    )
    array_activation, array_activation_scales = mxfp6.silu_and_mul_mxfp8_logical(
        array_gemm1_output
    )
    array_gemm2_output = torch.empty_like(gemm2_output)
    output = torch.empty_like(hidden_states)

    stages = {
        "quantize": lambda: mxfp6.quantize_mxfp8_logical(hidden_states),
        "permute": lambda: moe_permute(
            quantized,
            logical_scales,
            topk_ids,
            args.experts,
            args.experts,
        ),
        "pack_scales": lambda: mxfp6.pack_grouped_scales(
            permuted_scales, expert_offsets
        ),
        "gemm1": lambda: mxfp6.grouped_gemm_w6a8_out(
            gemm1_output,
            permuted,
            packed_scales,
            weights[0],
            scales[0],
            expert_offsets,
            scale_offsets,
        ),
        "silu_quant": lambda: mxfp6.silu_and_mul_mxfp8_grouped(
            gemm1_output, expert_offsets
        ),
        "gemm2": lambda: mxfp6.grouped_gemm_w6a8_out(
            gemm2_output,
            activation,
            activation_scales,
            weights[1],
            scales[1],
            expert_offsets,
            activation_scale_offsets,
        ),
        "array_gemm1": lambda: mxfp6.array_gemm_w6a8_out(
            array_gemm1_output,
            quantized,
            logical_scales,
            weights[0],
            scales[0],
            topk_ids,
        ),
        "array_silu": lambda: mxfp6.silu_and_mul_mxfp8_logical(array_gemm1_output),
        "array_gemm2": lambda: mxfp6.array_gemm_w6a8_out(
            array_gemm2_output,
            array_activation,
            array_activation_scales,
            weights[1],
            scales[1],
            topk_ids,
        ),
        "unpermute": lambda: moe_unpermute(
            output,
            gemm2_output,
            topk_weights,
            inverse_permutation,
            expert_offsets,
        ),
    }
    print("\nMXFP6 stage profile (CUDA graph, milliseconds):")
    for name, stage in stages.items():
        latency = _bench(
            stage,
            warmup=5,
            iterations=20,
            repeats=3,
            graph_calls=10,
        )
        print(f"  {name:12s} {latency:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--fp8-intermediate-size", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--graph-calls", type=int, default=0)
    parser.add_argument("--profile-batch", type=int)
    parser.add_argument("--tile-n", type=int, choices=[8, 16, 32, 64, 128])
    args = parser.parse_args()
    if args.fp8_intermediate_size is None:
        args.fp8_intermediate_size = args.intermediate_size

    import vllm._moe_C_stable_libtorch  # noqa: F401
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.manual_seed(2035)
    torch.cuda.set_device(0)
    init_workspace_manager(torch.device("cuda:0"))
    mxfp6.load_library()
    if args.tile_n is not None:
        grouped_gemm = mxfp6.grouped_gemm_w6a8_out

        def grouped_gemm_with_tile(*op_args, **op_kwargs):
            op_kwargs["tile_n"] = args.tile_n
            return grouped_gemm(*op_args, **op_kwargs)

        mxfp6.grouped_gemm_w6a8_out = grouped_gemm_with_tile
    dtype = torch.bfloat16
    moe_config, mx_kernel, mx_weights, fp8_weights, fp8_quant = _make_weights(
        args, dtype
    )

    print("batch  mxfp6_ms  fp8_block_ms  speedup")
    for batch in args.batch_sizes:
        hidden_states = torch.randn(
            (batch, args.hidden_size), device="cuda", dtype=dtype
        )
        token = torch.arange(batch, device="cuda", dtype=torch.int32)[:, None]
        lane = torch.arange(args.topk, device="cuda", dtype=torch.int32)[None, :]
        topk_ids = (token * args.topk + lane) % args.experts
        topk_weights = torch.rand(
            (batch, args.topk), device="cuda", dtype=torch.float32
        )
        topk_weights /= topk_weights.sum(dim=1, keepdim=True)

        def run_mxfp6():
            return mx_kernel.apply(
                hidden_states,
                mx_weights[0],
                mx_weights[1],
                topk_weights,
                topk_ids,
                activation=moe_config.activation,
                global_num_experts=args.experts,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

        def run_fp8_block():
            return fused_experts(
                hidden_states,
                fp8_weights[0],
                fp8_weights[1],
                topk_weights,
                topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=args.experts,
                quant_config=fp8_quant,
            )

        mx_ms = _bench(
            run_mxfp6,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            graph_calls=args.graph_calls,
        )
        fp8_ms = _bench(
            run_fp8_block,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            graph_calls=args.graph_calls,
        )
        print(f"{batch:5d}  {mx_ms:8.4f}  {fp8_ms:12.4f}  {fp8_ms / mx_ms:7.3f}x")

        if args.profile_batch == batch:
            _profile_mxfp6(
                hidden_states,
                topk_weights,
                topk_ids,
                mx_weights,
                (mx_kernel.fused_experts.w1_scale, mx_kernel.fused_experts.w2_scale),
                args,
            )


if __name__ == "__main__":
    main()
