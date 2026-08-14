# Compatibility

## Project scope

`mxfp6-sm120` is a source-built PyTorch CUDA extension for NVIDIA compute
capability 12.0. The public compatibility boundary is the Python and
`torch.ops.mxfp6` API in this repository. vLLM and SGLang are optional
benchmark/integration consumers, not package dependencies.

## Supported boundary

| Component | Current policy |
|---|---|
| GPU | Exactly SM120. Other compute capabilities fail closed at runtime. |
| OS | Linux. Other operating systems are not supported. |
| CUDA Toolkit | 12.8 or newer; the release is compiled for `sm_120a`. |
| PyTorch | CUDA-enabled PyTorch with FP8 dtype support. Build the extension against the exact PyTorch ABI used at runtime. |
| Python | 3.10 or newer. |
| CUTLASS | Pinned submodule commit `e6233cbac5d7c7a865c19c91cd684ceece19513c` plus the checked-in runtime patches. |
| Distribution | Source wheel only. No cross-PyTorch or cross-CUDA binary compatibility is promised. |

The build intentionally rejects an uninitialized or mismatched CUTLASS
checkout. The Python API checks CUDA placement, dtype, contiguity, shape and
SM120 capability before dispatch.

## Measured environment

The reviewed dense benchmark manifest was produced on an RTX 5090 with
PyTorch `2.11.0+cu130` and CUDA 13.0. The checked-in Qwen3.5 evidence includes
linear-layer kernels, complete MoE layers and pinned full-service comparisons
on RTX 5090 GPUs. Each manifest states its own model, runtime and measurement
boundary. None is a compatibility promise for arbitrary framework or model
revisions.

Two serving evidence sets are published. The public reproducer uses a
version-locked vLLM v0.25.1 image and a checked-in compatibility layer for
Qwen3.5-27B and Qwen3.5-35B-A3B. The internal reference uses an independently
optimized vLLM v0.25.1 environment and frozen workloads. Neither is a claim
that the standalone wheel is a general vLLM or SGLang backend.

## Runtime status

| Consumer | Available path | Boundary |
|---|---|---|
| PyTorch package | Dense and Qwen3.5 MoE operators | No model checkpoint loader |
| vLLM v0.25.1 | Public two-model TP2 reproducer under [`examples/vllm`](../examples/vllm/README.md) | Version-locked compatibility layer; not an upstream or general backend |
| SGLang | Not integrated | No checkpoint or execution adapter |

See [runtime integration](runtime-integration.md) for the current upstream
status and remaining integration work.

## Upgrade policy

Until a 1.0 release, CUDA operator schemas, workspace layouts and Qwen-specific
APIs may change between minor releases. Patch releases must remain compatible
within their minor line. Applications should pin both the package version and
the source commit.
