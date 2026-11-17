"""Prepared W6A8 calls with fixed storage and reusable CUTLASS parameters."""
import torch

from ._autotune_policy import W6A8Config
from ._loader import load_library

# Captured kernels reference the private workspace as well as the output.
# Keep captured owners alive for the process lifetime, including when callers
# discard the prepared callable before replaying a graph.
_GRAPH_OWNERS = {}


class PreparedW6A8:
    def __init__(self, native, device):
        self._native = native
        self.device = device
        self.config = W6A8Config(*native.config())
        self.backend = "aot"

    def __call__(self):
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                _GRAPH_OWNERS[id(self)] = self
            return self._native.run()


def prepare_w6a8(a, b, alpha=1.0, *, out_dtype=torch.float16,
                 config=None, pdl=False, backend="aot", source_root=None):
    """Prepare fixed operands; contents may change in place, storage may not.

    Returns a callable reusing one output and private workspace. Prepare before
    capture. Use independent objects for concurrent streams; captured objects
    are retained for process lifetime to protect graph-owned device pointers.
    """
    from .ops import (MXFP8Tensor, PackedMXFP6Tensor, _validate_output_dtype,
                      _needs_w6a8_autotune, _ensure_w6a8_autotuned)
    if backend not in ("aot", "jit"):
        raise ValueError("backend must be aot or jit")
    _validate_output_dtype(out_dtype)
    if not isinstance(a, MXFP8Tensor) or not isinstance(b, PackedMXFP6Tensor):
        raise TypeError("prepare_w6a8 expects MXFP8Tensor A and PackedMXFP6Tensor B")
    if a.k != b.k or a.device != b.device:
        raise ValueError("operands must have matching K and CUDA device")
    if config is not None and not isinstance(config, W6A8Config):
        raise TypeError("config must be a W6A8Config")
    if getattr(torch.compiler, "is_compiling", lambda: False)():
        raise RuntimeError("prepare_w6a8 must run before torch.compile")
    with torch.cuda.device(a.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare_w6a8 must run before CUDA Graph capture")
        load_library()
        if config is None and _needs_w6a8_autotune(a.device, a.rows, b.rows, a.k, out_dtype):
            _ensure_w6a8_autotuned(a.values, b.values, a.scales, b.scales,
                                 a.rows, b.rows, a.k, out_dtype=out_dtype)
        config = config or W6A8Config(-1, 1, 0)
        native = torch.ops.mxfp6.prepare_w6a8(
            a.values, b.values, a.scales, b.scales, a.rows, b.rows, a.k,
            alpha, config.config_id, config.swizzle, config.raster_order,
            out_dtype, pdl)
        run = PreparedW6A8(native, a.device)
        if backend == "jit":
            from ._w6a8_jit import bind
            run._compiled = bind(native, run.config, out_dtype, a.device, source_root)
            run.backend = "jit"
        return run
