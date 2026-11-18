"""Compare compact routing, expert arithmetic and changing-input graph replay."""

from dataclasses import replace

import pytest
import torch

import mxfp6


@pytest.fixture(scope="module")
def weights():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    generator = torch.Generator(device="cuda").manual_seed(735)
    options = dict(device="cuda", dtype=torch.uint8)
    return (
        torch.randint(256, (256, 512, 1536), generator=generator, **options),
        torch.full((256, 512 * 64), 119, **options),
        torch.randint(256, (256, 2048, 192), generator=generator, **options),
        torch.full((256, 2048 * 8), 119, **options),
    )


def _workspace(hidden):
    shapes = mxfp6.qwen35_grouped_workspace_shapes(len(hidden))
    output = torch.empty_like(hidden)
    first, second = [hidden.new_empty(shape) for shape in shapes]
    return mxfp6.Qwen35GroupedWorkspace.from_storage(output, first, second), output


def _generic_router(logits):
    # Spell out FP32 addition order. Each lane sums eight adjacent logits,
    # then performs the full-warp XOR reduction before top-8 renormalization.
    values = logits.float()
    probabilities = (values - values.amax(dim=1, keepdim=True)).exp()
    lanes = probabilities.reshape(len(logits), 32, 8)
    lane_sums = lanes[:, :, 0]
    for column in range(1, 8):
        lane_sums = lane_sums + lanes[:, :, column]
    lane_ids = torch.arange(32, device=logits.device)
    for mask in (16, 8, 4, 2, 1):
        lane_sums = lane_sums + lane_sums[:, lane_ids ^ mask]
    probabilities = (lanes * lane_sums.reciprocal().unsqueeze(-1)).flatten(1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    ids = probabilities.argsort(dim=1, descending=True, stable=True)[:, :8].contiguous()
    selected = probabilities.gather(1, ids)
    total = selected[:, 0]
    for slot in range(1, 8):
        total = total + selected[:, slot]
    inverse = torch.where(total > 0, total.reciprocal(), 0.0)
    return ids.int(), selected * inverse[:, None]


def _route_scales(scales, ids, inverse, offsets, scale_offsets):
    experts = ids.flatten().long()
    row = (scale_offsets[experts] + inverse - offsets[experts])[:, None]
    group = torch.arange(8, device=ids.device)[None, :]
    indices = (row // 128 * 2 + group // 4) * 512 + row % 32 * 16 + row % 128 // 32 * 4
    return scales.flatten()[indices + group % 4]


def _assert_matches_grouped(workspace, output, hidden, logits, weights):
    ids, topk = _generic_router(logits)
    w1, s1, w2, s2 = weights
    quantized, logical = mxfp6.quantize_mxfp8_logical(hidden)
    permuted = torch.empty_like(workspace.permuted)
    packed, offsets, scale_offsets, inverse = mxfp6.route_mxfp8_out(
        quantized, logical, ids, None, 256, permuted
    )
    gate_up = hidden.new_empty((ids.numel(), 512))
    mxfp6.grouped_gemm_w6a8_out(gate_up, permuted, packed, w1, s1, offsets, scale_offsets)
    activated, scales, _ = mxfp6.silu_and_mul_mxfp8_grouped(gate_up, offsets, scale_offsets)
    routed = torch.empty_like(workspace.routed_output)
    mxfp6.grouped_gemm_w6a8_out(routed, activated, scales, w2, s2, offsets, scale_offsets)
    reference = torch.empty_like(output)
    mxfp6.moe_reduce_out(reference, routed, topk, inverse)

    # Atomic placement can reorder rows within an expert. Compare each token/slot.
    actual_rows = workspace.inverse_permutation.flatten().long()
    ref_rows = inverse.flatten().long()
    route_ids = torch.arange(ids.numel(), device=ids.device)
    assert torch.equal(actual_rows.sort().values, route_ids)
    assert torch.equal(workspace.source_tokens[actual_rows], (route_ids // 8).int())
    assert torch.equal(workspace.expert_offsets, offsets)
    assert torch.equal(workspace.scale_offsets, scale_offsets)
    assert torch.equal(workspace.topk_ids, ids)
    assert torch.equal(workspace.topk_weights.view(torch.uint8), topk.view(torch.uint8))
    assert torch.equal(workspace.quantized, quantized.view(torch.uint8))
    assert torch.equal(workspace.input_scales, logical)
    assert torch.equal(workspace.activated[actual_rows], activated.view(torch.uint8)[ref_rows])
    assert torch.equal(
        _route_scales(workspace.activated_scales, ids, actual_rows, offsets, scale_offsets),
        _route_scales(scales, ids, ref_rows, offsets, scale_offsets),
    )
    assert torch.equal(
        workspace.routed_output[actual_rows].view(torch.uint8), routed[ref_rows].view(torch.uint8)
    )
    assert torch.equal(output.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("rows", [97, 160])
def test_compact_eager_and_changing_graph(weights, rows):
    generator = torch.Generator(device="cuda").manual_seed(rows)
    hidden = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16, generator=generator)
    logits = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    workspace, output = _workspace(hidden)

    def run():
        mxfp6.qwen35_compact_gemm_out(workspace, hidden, logits, *weights)
        mxfp6.qwen35_grouped_reduce_out(workspace, output)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        _assert_matches_grouped(workspace, output, hidden, logits, weights)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for concentrated in (True, False):
            hidden.normal_(generator=generator)
            logits.normal_(generator=generator)
            if concentrated:
                logits[:, 8:].fill_(-128)  # Empty experts and many tiles per hot expert.
            logits[0].zero_()  # Equal weights choose experts 0 through 7.
            logits[1, 3] = float("inf")
            workspace.routed_output.fill_(1)
            graph.replay()
            _assert_matches_grouped(workspace, output, hidden, logits, weights)
    stream.synchronize()


def test_compact_contract(weights):
    hidden = torch.empty(97, 2048, device="cuda", dtype=torch.bfloat16)
    logits = hidden.new_empty((97, 256))
    workspace, _ = _workspace(hidden)
    for rows in (96, 161):
        with pytest.raises(RuntimeError, match="97 to 160"):
            mxfp6.qwen35_compact_gemm_out(workspace, hidden.new_empty((rows, 2048)), logits, *weights)
    with pytest.raises(RuntimeError, match="routed_logits"):
        mxfp6.qwen35_compact_gemm_out(workspace, hidden, logits.float(), *weights)
    with pytest.raises(RuntimeError, match="w1 requires"):
        mxfp6.qwen35_compact_gemm_out(workspace, hidden, logits, weights[0][:-1], *weights[1:])
    short = replace(workspace, activated_scales=workspace.activated_scales[:1])
    with pytest.raises(RuntimeError, match="grouped MXFP8 scale storage"):
        mxfp6.qwen35_compact_gemm_out(short, hidden, logits, *weights)
