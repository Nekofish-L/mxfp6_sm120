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

Current Dense, complete-MoE-layer and full-service results were produced on RTX
5090 GPUs with PyTorch `2.11.0+cu130` and CUDA 13.0. The latest service artifact
uses an internally optimized vLLM 0.25.1 Champion and covers Qwen3.5-27B and
Qwen3.5-35B-A3B at concurrency 1 through 32. Each artifact records its model,
runtime and measurement boundary.

The public reproducer uses a separate, version-locked vLLM v0.28.0 image and a
checked-in two-model adapter. It is the independently runnable setup. The
Champion result is the broader performance reference. Neither environment
makes the standalone wheel a general vLLM or SGLang backend.

## Runtime status

| Consumer | Available path | Boundary |
|---|---|---|
| PyTorch package | Dense and Qwen3.5 MoE operators | No model checkpoint loader |
| vLLM v0.28.0 | Public two-model TP2 reproducer under [`examples/vllm`](../examples/vllm/README.md) | Version-locked adapter for the tested profiles |
| Internal Champion | Two-model TP2 concurrency sweep | Recorded performance environment, not a distributable runtime |
| SGLang | Not integrated | No checkpoint or execution adapter |

See [runtime integration](runtime-integration.md) for the current upstream
status and remaining integration work.

## Upgrade policy

Until a 1.0 release, CUDA operator schemas, workspace layouts and Qwen-specific
APIs may change between minor releases. Patch releases must remain compatible
within their minor line. Applications should pin both the package version and
the source commit.
