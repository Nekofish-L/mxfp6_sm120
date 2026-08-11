from pathlib import Path

import torch
import torch.nn.functional as F

import mxfp6
from benchmarks.benchmark_qwen35_moe_layer import load_mx_layer


def capture(function):
    for _ in range(10):
        output = function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    return graph, output


def benchmark(graph: torch.cuda.CUDAGraph) -> float:
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(500):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 500)
    return sorted(samples)[len(samples) // 2]


def main() -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    weights = load_mx_layer(
        Path("/data/models/Qwen3.5-35B-A3B-MXFP6"),
        0,
        2,
        device,
    )
    for batch in (20, 96):
        hidden = torch.randn((batch, 2048), device=device, dtype=torch.bfloat16)
        logits = F.linear(hidden, weights.combined_gate[:256]).float()
        probabilities = torch.softmax(logits, dim=-1)
        topk_weights, topk_ids = torch.topk(probabilities, 8, dim=-1)
        topk_ids = topk_ids.to(torch.int32)
        first_shape, second_shape = mxfp6.qwen35_grouped_workspace_shapes(batch)
        first = torch.empty(first_shape, device=device, dtype=torch.bfloat16)
        second = torch.empty(second_shape, device=device, dtype=torch.bfloat16)
        output = torch.empty((batch, 2048), device=device, dtype=torch.bfloat16)
        workspace = mxfp6.Qwen35GroupedWorkspace.from_storage(
            output,
            first,
            second,
        )
        workspace.grouped_input_scales.fill_(0x7F)
        workspace.activated_scales.fill_(0x7F)
        def grouped():
            return mxfp6.qwen35_grouped_moe_out(
                workspace,
                output,
                hidden,
                weights.w1[:256],
                weights.w1_scale[:256],
                weights.w2[:256],
                weights.w2_scale[:256],
                topk_weights,
                topk_ids,
            )

        generic_permuted = torch.empty(
            (batch * 8, 2048), device=device, dtype=torch.float8_e4m3fn
        )
        generic_gemm1 = torch.empty(
            (batch * 8, 512), device=device, dtype=torch.bfloat16
        )
        generic_gemm2 = torch.empty(
            (batch * 8, 2048), device=device, dtype=torch.bfloat16
        )
        generic_output = torch.empty_like(output)

        def generic():
            quantized, logical_scales = mxfp6.quantize_mxfp8_logical(hidden)
            packed, offsets, scale_offsets, inverse = mxfp6.route_mxfp8_out(
                quantized,
                logical_scales,
                topk_ids,
                None,
                256,
                generic_permuted,
            )
            mxfp6.grouped_gemm_w6a8_out(
                generic_gemm1,
                generic_permuted,
                packed,
                weights.w1[:256],
                weights.w1_scale[:256],
                offsets,
                scale_offsets,
            )
            activated, scales, activated_offsets = (
                mxfp6.silu_and_mul_mxfp8_grouped(
                    generic_gemm1,
                    offsets,
                    scale_offsets,
                )
            )
            mxfp6.grouped_gemm_w6a8_out(
                generic_gemm2,
                activated,
                scales,
                weights.w2[:256],
                weights.w2_scale[:256],
                offsets,
                activated_offsets,
            )
            mxfp6.moe_reduce_out(
                generic_output,
                generic_gemm2,
                topk_weights,
                inverse,
            )
            return generic_output

        generic_graph, generic_result = capture(generic)
        grouped_graph, grouped_result = capture(grouped)
        torch.cuda.synchronize()
        if not torch.isfinite(grouped_result).all():
            raise AssertionError(f"non-finite output for B={batch}")
        difference = grouped_result.float() - generic_result.float()
        relative_rms = (
            difference.square().mean().sqrt()
            / generic_result.float().square().mean().sqrt()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            grouped_result.float().flatten(),
            generic_result.float().flatten(),
            dim=0,
        ).item()
        generic_ms = benchmark(generic_graph)
        grouped_ms = benchmark(grouped_graph)
        print(
            f"B={batch} generic={generic_ms:.4f}ms grouped={grouped_ms:.4f}ms "
            f"speedup={generic_ms / grouped_ms:.3f}x rel_rms={relative_rms:.4f} "
            f"cos={cosine:.4f} generic_rms="
            f"{generic_result.float().square().mean().sqrt().item():.6f} "
            f"grouped_rms={grouped_result.float().square().mean().sqrt().item():.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
