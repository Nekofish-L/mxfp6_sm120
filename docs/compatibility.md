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
PyTorch `2.11.0+cu130` and CUDA 13.0. The checked-in Qwen3.5 results are
kernel/layer measurements on RTX 5090 GPUs. They are evidence for the declared
operators and shapes, not a compatibility promise for arbitrary framework or
model revisions.

Some development measurements used a locally patched vLLM environment. Do not
interpret them as results from an unmodified community vLLM release. A runtime
integration must record its exact vLLM or SGLang commit, extension wheel hash,
checkpoint revision and launch command.

## Runtime status

| Consumer | Dense operator | Qwen3.5 MoE layer | Checkpoint loader | Supported serving integration |
|---|---:|---:|---:|---:|
| PyTorch package | Yes | Experimental | No | Not applicable |
| vLLM | Benchmark use only | Benchmark use only | No | No |
| SGLang | Not integrated | Not integrated | No | No |

See [runtime integration](runtime-integration.md) for the work required to
change the last column.

## Upgrade policy

Until a 1.0 release, CUDA operator schemas, workspace layouts and Qwen-specific
APIs may change between minor releases. Patch releases must remain compatible
within their minor line. Applications should pin both the package version and
the source commit.
