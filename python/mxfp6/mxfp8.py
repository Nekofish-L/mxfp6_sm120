"""Native W8A8 API, following the MXFP6 call and workspace conventions."""

from __future__ import annotations
from typing import NamedTuple
import torch

from ._loader import load_mxfp8_library as load_library
from ._workspace import WorkspaceAPI
from .ops import (
    MXFP8Tensor,
    quantize_activation,
    quantize_mxfp8,
    pack_scales,
    unpack_scales,
)


class W8A8Config(NamedTuple):
    config_id: int
    splits: int = 1
    swizzle: int = 1
    sms: int = 0


def bf16_ba_m16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute BF16 X @ W.T for contiguous X[16,2048], W[32,2048] on SM120.

    Inputs require 16-byte-aligned storage. This dedicated BA projection does
    not quantize inputs; FP32 MMA accumulation is rounded once to BF16.
    Warm this call before CUDA graph capture, then mutate inputs in place.
    """
    load_library()
    return torch.ops.mxfp8_sm120.bf16_ba_m16(x, weight)


def select_config(m: int, n: int, k: int) -> W8A8Config:
    """Inspect the configuration selected by the compiled C++ interval policy."""
    load_library()
    return W8A8Config(*torch.ops.mxfp8_sm120.select_config(m, n, k))


def available_configs(*, stream_k_only=False, backend="aot"):
    """List tactic IDs, not compiler readiness or arbitrary scheduler support.

    The JIT portfolio requires splits/swizzle/sms=1/1/0. Default AOT discovery
    is unchanged; auto exposes the union without importing the JIT runtime.
    """
    if backend not in ("aot", "auto", "jit"):
        raise ValueError("backend must be 'aot', 'auto' or 'jit'")
    jit = () if backend == "aot" or stream_k_only else available_jit_configs()
    if backend == "jit":
        return jit
    load_library()
    native = tuple(torch.ops.mxfp8_sm120.tactics(stream_k_only))
    return tuple(sorted(set(native).union(jit))) if backend == "auto" else native


def _check_output(alpha, out_dtype):
    if alpha != 1.0:
        raise ValueError("MXFP8 kernels currently support alpha=1.0")
    if out_dtype != torch.bfloat16:
        raise ValueError("MXFP8 kernels currently produce torch.bfloat16")


def _operands(a, b):
    if not isinstance(a, MXFP8Tensor) or not isinstance(b, MXFP8Tensor):
        raise TypeError("a and b must be MXFP8Tensor instances")
    if a.k != b.k or a.device != b.device:
        raise ValueError("a and b must have matching K and CUDA device")
    return a.dequantized_values(), b.dequantized_values(), a.scales, b.scales


def _config(config):
    if config is None:
        return (-1, 1, 1, 0)
    if not isinstance(config, W8A8Config):
        raise TypeError("config must be a W8A8Config instance")
    return tuple(config)


def gemm_w8a8(
    a: MXFP8Tensor,
    b: MXFP8Tensor,
    alpha=1.0,
    *,
    out_dtype=torch.bfloat16,
    config=None,
    out=None,
    workspace=None,
    use_pdl=False,
):
    """Compute prequantized MXFP8(A) @ MXFP8(B).T through native dispatch."""
    _check_output(alpha, out_dtype)
    operands = _operands(a, b)
    load_library()
    return torch.ops.mxfp8_sm120.gemm(*operands, out, workspace, *_config(config), use_pdl)


def gemm_from_float(
    a: torch.Tensor, b: MXFP8Tensor, alpha=1.0, *, out_dtype=torch.bfloat16, use_pdl=False
):
    """Quantize FP16/BF16 A with the shared native quantizer, then run W8A8."""
    _check_output(alpha, out_dtype)
    if not isinstance(b, MXFP8Tensor):
        raise TypeError("b must be an MXFP8Tensor instance")
    if a.ndim != 2 or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("a must be an FP16/BF16 matrix")
    if a.shape[1] != b.k or a.device != b.device:
        raise ValueError("a and b must have matching K and CUDA device")
    from ._loader import load_library as load_quantizer

    load_quantizer()
    load_library()
    return torch.ops.mxfp8_sm120.gemm_from_float(a, b.dequantized_values(), b.scales, use_pdl)


def gemm(
    a: torch.Tensor | MXFP8Tensor,
    b: MXFP8Tensor,
    alpha=1.0,
    *,
    out_dtype=torch.bfloat16,
    use_pdl=False,
):
    """Compute A @ B.T; floating activations are quantized in the native path."""
    if isinstance(a, torch.Tensor):
        return gemm_from_float(a, b, alpha, out_dtype=out_dtype, use_pdl=use_pdl)
    return gemm_w8a8(a, b, alpha, out_dtype=out_dtype, use_pdl=use_pdl)


def gemm_packed(
    a,
    b,
    sfa,
    sfb,
    m,
    n,
    k,
    alpha=1.0,
    *,
    out_dtype=torch.bfloat16,
    config=None,
    out=None,
    workspace=None,
    use_pdl=False,
):
    """Low-level A @ B.T for byte-aligned E4M3 values and packed scales."""
    _check_output(alpha, out_dtype)
    for value in (a, b):
        if value.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise TypeError("MXFP8 values must have uint8 or float8_e4m3fn dtype")
    load_library()
    av = a.view(m, k).view(torch.float8_e4m3fn)
    bv = b.view(n, k).view(torch.float8_e4m3fn)
    return torch.ops.mxfp8_sm120.gemm(
        av, bv, sfa, sfb, out, workspace, *_config(config), use_pdl
    )


def prepare(
    a: MXFP8Tensor, b: MXFP8Tensor, *, out_dtype=torch.bfloat16, config=None, out=None,
    use_pdl=False, backend="aot", source_root=None,
):
    """Prepare one selected configuration for graph replay.

    Inputs and scales may change in place. Concurrent calls need independent
    prepared objects. ``aot`` preserves the default native path. ``auto`` uses
    AOT when the selected tactic is present, otherwise JIT; ``jit`` forces JIT.
    JIT supports only migrated tactics and must be prepared before capture.
    Backend selection never substitutes a different tactic or retries failures.
    ``source_root`` supplies the matching source checkout for JIT compilation.
    """
    if backend not in ("aot", "auto", "jit"):
        raise ValueError("backend must be 'aot', 'auto' or 'jit'")
    _check_output(1.0, out_dtype)
    operands = _operands(a, b)
    requested = _config(config)
    if backend != "aot":
        selected = select_config(a.rows, b.rows, a.k) if requested[0] == -1 else config
        if backend == "jit" or selected.config_id not in available_configs():
            from ._mxfp8_jit import prepare as prepare_backend
            return prepare_backend(a, b, config=selected, out=out,
                                   use_pdl=use_pdl, source_root=source_root)
        requested = tuple(selected)
    load_library()
    output, workspace, selected = torch.ops.mxfp8_sm120.prepare(
        *operands, out, *requested
    )
    selected = W8A8Config(*selected)

    def run():
        return torch.ops.mxfp8_sm120.gemm(*operands, output, workspace, *selected, use_pdl)

    run.config = selected
    run.backend = "aot"
    return run


def available_jit_configs():
    """List described JIT tactics without loading CUDA or the optional compiler.

    This reports the migrated portfolio, not toolchain readiness or AOT
    availability. Execution still requires the optional source/toolchain setup.
    """
    from ._mxfp8_config import configurations, load_manifest
    return tuple(c['tactic'] for c in configurations(load_manifest()))


def precompile_jit(config: W8A8Config, k: int, *, device=None, source_root=None):
    """Precompile one migrated tactic before capture; return an opaque cached handle.

    Opt-in source-checkout backend requiring NVCC 12.9+ and DeepJIT. Only the
    88 migrated cp.async tactics with splits/swizzle/sms=1/1/0 are supported.
    Source/compiler settings are fixed for the process lifetime. Handles are
    retained until process exit, including after prepared callables are deleted.
    """
    from ._mxfp8_jit import precompile
    return precompile(config, k, device=device, source_root=source_root)


def prepare_jit(a: MXFP8Tensor, b: MXFP8Tensor, *, config: W8A8Config,
                out=None, use_pdl=False, source_root=None):
    """Prepare an explicit JIT tactic with private BF16 output and no workspace.

    Call before CUDA Graph capture; replay performs no compilation or cache I/O.
    Inputs may change in place but must not be resized. Concurrent calls need
    independent prepared objects. The default AOT prepare/dispatch is unaffected.
    """
    if not isinstance(config, W8A8Config):
        raise TypeError("JIT requires an explicit W8A8Config")
    return prepare(a, b, config=config, out=out, use_pdl=use_pdl,
                   backend="jit", source_root=source_root)


def allocate_workspace(a: MXFP8Tensor, b: MXFP8Tensor, *, config=None, out=None):
    """Allocate private native workspace for explicit low-level launches."""
    operands = _operands(a, b)
    load_library()
    _, workspace, _ = torch.ops.mxfp8_sm120.prepare(*operands, out, *_config(config))
    return workspace


def warmup(a, b, *, out_dtype=torch.bfloat16, iterations=3, use_pdl=False):
    """Warm the native path before capture; collect layouts when planning."""
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations <= 0
    ):
        raise ValueError("iterations must be a positive integer")
    for _ in range(iterations):
        gemm(a, b, out_dtype=out_dtype, use_pdl=use_pdl)
    torch.cuda.synchronize(a.device)
    m, k = a.shape
    return select_config(m, b.rows, k)


_workspace = WorkspaceAPI("mxfp8_sm120", load_library)
begin_workspace_planning = _workspace.begin_workspace_planning
finalize_workspace_planning = _workspace.finalize_workspace_planning
workspace_stats = _workspace.workspace_stats
workspace_barriers_zero = _workspace.workspace_barriers_zero

__all__ = [
    "MXFP8Tensor",
    "W8A8Config",
    "quantize_mxfp8",
    "quantize_activation",
    "pack_scales",
    "unpack_scales",
    "bf16_ba_m16",
    "gemm",
    "gemm_w8a8",
    "gemm_from_float",
    "gemm_packed",
    "prepare",
    "prepare_jit",
    "precompile_jit",
    "available_jit_configs",
    "warmup",
    "select_config",
    "available_configs",
    "allocate_workspace",
    "load_library",
    "begin_workspace_planning",
    "finalize_workspace_planning",
    "workspace_stats",
    "workspace_barriers_zero",
]
