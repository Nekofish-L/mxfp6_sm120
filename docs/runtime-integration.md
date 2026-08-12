# vLLM and SGLang integration status

## Current verdict

This repository is ready to be reviewed as an SM120 kernel package and layer
benchmark reference. It is not yet a drop-in serving backend for vLLM or
SGLang. The missing work is at the checkpoint, loader, model and runtime
boundaries rather than in installation prose.

The repository also publishes pinned Qwen3.5-27B and Qwen3.5-35B-A3B serving
comparisons from one internally maintained vLLM environment. Those experiments
demonstrate end-to-end potential and preserve their measurement contracts; they
do not turn the private research adapter into a supported public integration.

## vLLM

Current vLLM main provides an `MxFp6LinearKernel` backend interface and an
out-of-tree kernel registration mechanism. Its own CPU test describes software
emulation as the only available implementation at the time of this release.
That interface is the preferred dense integration point for this project.

A production contribution still needs:

1. an SM120 backend implementing support checks, weight post-processing and
   `apply_weights` without importing private benchmark code;
2. a versioned checkpoint mapping and loader path;
3. model-layer selection for dense projections and a separate MoE design;
4. custom-op fake/meta behavior where required by compilation and CUDA Graphs;
5. exact unsupported-platform and unsupported-checkpoint behavior;
6. upstream unit tests plus public full-model prefill/decode, TP and graph
   validation.

The Qwen3.5 TP2 adapter used during development is intentionally not shipped
in this wheel. It is tied to a particular runtime tree and has not been reduced
to a stable vLLM extension contract.

## SGLang

SGLang exposes its own `QuantizationConfig`, linear and fused-MoE integration
contracts. It currently has no MXFP6 backend in its quantization registry.
Integration therefore requires a separate design and PR covering:

1. quantization config recognition and capability gating;
2. serialized weight creation/loading and post-load packing;
3. linear and Qwen3.5 fused-MoE dispatch;
4. CUDA Graph/custom-op integration and fallback behavior;
5. SGLang-native correctness, accuracy and serving benchmarks.

The vLLM adapter must not be copied into SGLang: their model, MoE and graph
interfaces differ.

## Required evidence for an upstream runtime PR

- a public, checksummed checkpoint conversion or conversion recipe;
- a pinned CUDA, PyTorch, runtime and model compatibility matrix;
- package build/install smoke tests in a clean Linux environment;
- config detection, tensor mapping, packing and dispatch unit tests;
- full-model prefill and decode under CUDA Graphs and TP2;
- public quality results with a stated acceptance threshold;
- end-to-end serving throughput, TTFT and TPOT against a reproducible baseline;
- raw commands and artifacts, with every claim limited to its measured scope.

Kernel and layer speedups alone are useful evidence, but they do not satisfy
this runtime acceptance bar.

## Recommended sequence

1. Freeze checkpoint format v1 and add a converter plus CPU metadata validator
   in this repository.
2. Implement the vLLM dense backend against its public MXFP6 kernel interface,
   then add the narrow Qwen3.5 MoE path.
3. Validate one model revision and SM120 TP2 configuration end to end before
   proposing upstream support.
4. Implement and validate SGLang independently after the format stabilizes.

## Upstream references

- [vLLM MXFP6 kernel interface](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/mxfp6/base.py)
- [vLLM kernel selection test](https://github.com/vllm-project/vllm/blob/main/tests/kernels/quantization/test_mxfp6_kernel_selection.py)
- [vLLM quantization configuration contract](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/base_config.py)
- [SGLang quantization configuration contract](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/base_config.py)
- [SGLang MXFP4 reference implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/mxfp4.py)
