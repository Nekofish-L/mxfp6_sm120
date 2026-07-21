"""Model-derived GEMM shapes shared by benchmark and autotuning tools."""

from __future__ import annotations


# Qwen3.5-27B, tensor parallel size 2. Each entry is (layer, N, K) for
# D[M, N] = A[M, K] @ B[N, K].T. Keeping the operator convention here avoids
# accidentally copying framework weight shapes in their transposed order.
QWEN35_27B_TP2_NK = (
    ("gdn_in_proj_qkvz", 8192, 5120),
    ("gdn_out_proj", 5120, 3072),
    ("full_attention_qkv_gate_proj", 7168, 5120),
    ("mlp_gate_up_proj", 17408, 5120),
    ("mlp_down_proj", 5120, 8704),
)

DEFAULT_BATCH_SIZES = (1, 16, 32, 64, 96, 512, 1024, 2048, 4096, 8192)
DEFAULT_SHAPES = tuple(
    (m, n, k)
    for m in DEFAULT_BATCH_SIZES
    for _, n, k in QWEN35_27B_TP2_NK
)
