"""Common Python interface to each native backend's Stream-K workspace pool."""

from __future__ import annotations
import torch


class WorkspaceAPI:
    def __init__(self, namespace, load_library):
        self.namespace = namespace
        self.load_library = load_library

    def _anchor(self, device):
        if device is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int) and not isinstance(device, bool):
            resolved = torch.device("cuda", device)
        else:
            resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(f"workspace device must be CUDA; got {resolved}")
        self.load_library()
        return torch.empty(0, device=resolved, dtype=torch.uint8)

    def begin_workspace_planning(self, device=None):
        """Collect layouts during eager warmup before freezing workspace capacity."""
        anchor = self._anchor(device)
        getattr(torch.ops, self.namespace).begin_workspace_planning(anchor)

    def finalize_workspace_planning(self, device=None):
        """Freeze capacity and allocate the current stream's workspace lane."""
        anchor = self._anchor(device)
        return dict(
            getattr(torch.ops, self.namespace).finalize_workspace_planning(anchor)
        )

    def workspace_stats(self, device=None):
        """Return layout, capacity, stream-lane and launch counters."""
        anchor = self._anchor(device)
        return dict(getattr(torch.ops, self.namespace).workspace_stats(anchor))

    def workspace_barriers_zero(self, device=None):
        """Synchronize and check the barrier tails of every frozen stream lane."""
        anchor = self._anchor(device)
        return bool(getattr(torch.ops, self.namespace).workspace_barriers_zero(anchor))
