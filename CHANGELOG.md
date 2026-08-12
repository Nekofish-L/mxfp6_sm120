# Changelog

## 0.1.1 - 2026-08-12

First GitHub release, covering both dense and Qwen3.5 MoE execution on SM120.

### Added

- native W6A8 dense GEMM with packed E3M2 weights and dynamic E4M3 activations;
- deterministic small-shape dispatch, deployment-safe autotune cache policy
  and persistent Stream-K workspace planning;
- allocation-free Qwen3.5-35B-A3B MoE layer schedules for TP2 CUDA Graph use;
- reviewed dense and MoE benchmark manifests;
- explicit compatibility, checkpoint-format and vLLM/SGLang integration
  documentation;
- source metadata checks and third-party notices.

### Compatibility boundary

- exact SM120 only;
- source build against the active PyTorch CUDA ABI;
- no stable serialized checkpoint format or runtime loader yet;
- no supported vLLM or SGLang serving integration in this release.
