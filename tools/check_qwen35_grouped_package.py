#!/usr/bin/env python3
"""Check the package-owned Qwen3.5 grouped schedule on real layer weights."""

import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import mxfp6
from benchmark_qwen35_moe_layer import (
    HIDDEN_SIZE,
    NUM_EXPERTS,
    load_mx_layer,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)


def make_workspace(batch: int, device: torch.device):
    output = torch.empty((batch, HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
    first_shape, second_shape = mxfp6.qwen35_grouped_workspace_shapes(batch)
    first = torch.empty(first_shape, device=device, dtype=torch.bfloat16)
    second = torch.empty(second_shape, device=device, dtype=torch.bfloat16)
    workspace = mxfp6.Qwen35GroupedWorkspace.from_storage(output, first, second)
    return workspace, output


def main() -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    mxfp6.load_library()
    weights = load_mx_layer(
        Path("/data/models/Qwen3.5-35B-A3B-MXFP6"),
        0,
        2,
        device,
    )
    for batch in (16, 32, 64, 96):
        torch.manual_seed(1000 + batch)
        hidden = torch.randn((batch, HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
        logits = F.linear(hidden, weights.combined_gate[:NUM_EXPERTS])

        reference_workspace, reference = make_workspace(batch, device)
        probabilities = torch.softmax(logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(probabilities, 8, dim=-1)
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        vllm_weights, vllm_ids, _ = fused_topk(
            hidden,
            logits,
            8,
            renormalize=True,
        )
        os.environ["MXFP6_GROUPED_REDUCE_X4"] = "0"
        mxfp6.qwen35_grouped_moe_out(
            reference_workspace,
            reference,
            hidden,
            weights.w1[:NUM_EXPERTS],
            weights.w1_scale[:NUM_EXPERTS],
            weights.w2[:NUM_EXPERTS],
            weights.w2_scale[:NUM_EXPERTS],
            topk_weights,
            topk_ids.to(torch.int32),
        )

        workspace, actual = make_workspace(batch, device)
        os.environ["MXFP6_GROUPED_REDUCE_X4"] = "1"
        mxfp6.qwen35_grouped_moe_logits_out(
            workspace,
            actual,
            hidden,
            logits,
            weights.w1[:NUM_EXPERTS],
            weights.w1_scale[:NUM_EXPERTS],
            weights.w2[:NUM_EXPERTS],
            weights.w2_scale[:NUM_EXPERTS],
            renormalize=True,
        )
        torch.cuda.synchronize()

        if batch in (32, 64):
            activated = workspace.activated.view(torch.float8_e4m3fn).float()
            finite = torch.isfinite(activated)
            bad_rows = (~finite).any(dim=1).nonzero().flatten()
            bad_columns = (~finite).any(dim=0).nonzero().flatten()
            print(
                f"  activated_finite={finite.float().mean().item():.6f} "
                f"bad_rows={bad_rows[:16].tolist()} "
                f"bad_columns={bad_columns[:32].tolist()} "
                f"routed_finite={torch.isfinite(workspace.routed_output).float().mean().item():.6f}"
            )

        difference = actual.float() - reference.float()
        relative_rms = float(
            difference.square().mean().sqrt()
            / reference.float().square().mean().sqrt()
        )
        cosine = float(
            F.cosine_similarity(actual.float().flatten(), reference.float().flatten(), dim=0)
        )
        ids_equal = bool(torch.equal(workspace.topk_ids, vllm_ids))
        weights_error = float(
            (workspace.topk_weights - vllm_weights).abs().max()
        )
        digest = hashlib.sha256(
            actual.view(torch.uint16).cpu().numpy().tobytes()
        ).hexdigest()[:16]
        print(
            f"B{batch}: rel_rms={relative_rms:.6g} cosine={cosine:.8f} "
            f"ids_equal={ids_equal} topk_max_abs={weights_error:.6g} "
            f"sha256={digest}"
        )


if __name__ == "__main__":
    main()
