"""CPU-only proof that the accepted BA kernel was copied without changes."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# SHA256 of the exact accepted prototype region from "// Each CTA owns" to
# immediately before its extern "C" prepare(), including trailing newlines.
ACCEPTED_KERNEL_SHA256 = "4f05f357008163ff1430f8fa8bbc6fdf08c36cd2d9bbad311788519721ce7bba"


def test_accepted_kernel_source_is_unchanged():
    source = (ROOT / "csrc/mxfp8_bf16_ba.cu").read_text()
    kernel = "// Each CTA owns" + source.split("// Each CTA owns", 1)[1].split(
        "} // namespace", 1
    )[0]
    assert hashlib.sha256(kernel.encode()).hexdigest() == ACCEPTED_KERNEL_SHA256


