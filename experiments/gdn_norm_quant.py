"""Diagnostic adapter for the exact extension-owned GDN producer."""
import torch
from mxfp6.gdn import gated_norm_mxfp8


def produce(x, z, w, eps=1.e-6):
    rounded = torch.empty_like(x)
    q, scales = gated_norm_mxfp8(x, z, w, eps, rounded_out=rounded)
    return q, scales, rounded
