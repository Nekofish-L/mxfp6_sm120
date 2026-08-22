# Runtime integration

`mxfp6-sm120` is an optional CUDA operator package. It has no vLLM or SGLang
dependency and can be built and tested as a standalone PyTorch extension.

## vLLM

vLLM recognizes Quark/OCP MXFP6 checkpoints. Its current CUDA MXFP6 backend
uses software emulation when no native implementation is registered.

vLLM commit `e9d1398d9edfd90fcc1cf783805240e3effec013` (reviewed 2026-08-22)
exposes the two integration surfaces needed by this package:

| Model path | vLLM interface | mxfp6-sm120 implementation |
|---|---|---|
| Dense | `MxFp6LinearKernel` | Weight processing, capability checks and native W6A8 GEMM |
| Routed MoE | `FusedMoEExpertsModular` | Routing, W1, activation, W2, shared expert and reduction |

The Dense kernel registry already supports an optional out-of-tree
implementation and retains `EmulationMxfp6LinearKernel` as fallback. The MoE
integration must select the native experts implementation only for a recognized
Quark MXFP6 layout, supported Qwen3.5 geometry and SM120 device. Other cases
continue through vLLM's existing path.

The repository's [`examples/vllm`](../examples/vllm/README.md) directory is a
version-locked vLLM v0.25.1 reproducer. It loads and serves Qwen3.5-27B and
Qwen3.5-35B-A3B, including TP2 and CUDA Graph execution. The adapter proves the
two model mappings but is not intended for current-main submission.

The current performance and correctness evidence is in
[Benchmark methodology](benchmarks.md). Discussion with vLLM maintainers is
tracked in [issue #52347](https://github.com/vllm-project/vllm/issues/52347).

### Contribution scope

A current-main integration needs to preserve one backend across both model
paths while keeping the changes reviewable:

1. register the SM120 Dense implementation through `MxFp6LinearKernel`;
2. connect the same Quark MXFP6 format to the native routed-MoE experts path;
3. keep package import optional and retain emulation fallback;
4. test checkpoint loading, TP2, CUDA Graph capture and unsupported cases;
5. document installation and the tested model and hardware boundary.

Under this boundary, CUDA sources and GPU-specific tests remain with the
optional package and the vLLM change is limited to registration, dispatch and
fallback coverage.

## SGLang

SGLang has separate quantization, linear and fused-MoE contracts. It currently
has no MXFP6 entry in its quantization registry, so vLLM adapter code cannot be
reused directly. A SGLang integration would need its own config recognition,
weight loading, kernel registration, graph tests and fallback behavior.

The native operators and checkpoint data layout can be shared across runtimes.
The framework glue remains runtime specific.

## References

- [vLLM MXFP6 kernel interface](https://github.com/vllm-project/vllm/blob/e9d1398d9edfd90fcc1cf783805240e3effec013/vllm/model_executor/kernels/linear/mxfp6/base.py)
- [vLLM MXFP6 kernel selection tests](https://github.com/vllm-project/vllm/blob/e9d1398d9edfd90fcc1cf783805240e3effec013/tests/kernels/quantization/test_mxfp6_kernel_selection.py)
- [vLLM modular MoE design](https://github.com/vllm-project/vllm/blob/e9d1398d9edfd90fcc1cf783805240e3effec013/docs/design/fused_moe_modular_kernel.md)
- [vLLM Quark OCP MX scheme](https://github.com/vllm-project/vllm/blob/e9d1398d9edfd90fcc1cf783805240e3effec013/vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py)
- [SGLang quantization base](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/base_config.py)
