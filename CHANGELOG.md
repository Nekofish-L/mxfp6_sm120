# Changelog

## Unreleased

## 0.2.1 - 2026-08-24

### Added

- added a fixed-concurrency TP2 serving sweep for Qwen3.8-27B across FP8,
  MXFP6 and standard NVFP4 W4A4, including two service lifecycles per format;
- added machine-readable concurrency evidence and an absolute-throughput curve.

### Changed

- replaced the Qwen3.5-35B-A3B combined-weight B4 W1 global split-K reduction
  with a two-block cluster-local reduction; the measured internal integration
  combines it with the existing packed W2 path;
- updated the c4 full-service and production-layer artifacts. The new path is
  bitwise identical to the previous MXFP6 Champion at the tested operator
  boundary and improves c4 output throughput by 4.41% over that Champion.

### Documentation

- expanded the Qwen3.8-27B report with 256-record BF16-fidelity results,
  c1-c32 throughput, checkpoint size, model load and KV-cache capacity;
- reframed MXFP6 as a measured quality-throughput-memory operating point rather
  than a universal replacement for FP8 or NVFP4;
- identified the public BF16 and official FP8 baselines and tightened the
  reproducibility and claim boundaries.

## 0.2.0 - 2026-08-22

### Added

- added a version-locked vLLM v0.25.1 reproducer for Qwen3.5-27B Dense and
  Qwen3.5-35B-A3B routed MoE, including public conversion, loading and serving
  checks;
- added systematic Dense, MoE layer and TP2 full-service benchmark artifacts,
  plus a reproducible concurrency-sweep plot;
- added an occupancy-aware cooperative W1 schedule for Qwen3.5 MoE batches 4-8
  on lower-SM-count SM120 devices.

### Changed

- extended exact Dense dispatch for the measured M24 Qwen3.5 shapes;
- reorganized runtime, checkpoint, integration and benchmark documentation
  around the two validated Qwen3.5 execution paths;
- made the public reproducer build its runtime and converter from their pinned
  source revisions.

### Fixed

- replaced profiler-backed autotune timing with CUDA Graph-safe event timing;
- fixed the large-token split-K MoE path covered by the 5,000-token regression;
- tightened packaging and Docker provenance so source artifacts are built from
  the checked-out revision.

### Compatibility boundary

- native kernels remain specific to SM120 and are built against the active
  PyTorch CUDA ABI;
- the vLLM integration is a model-scoped, version-locked reproducer, not an
  upstream vLLM backend or a stable cross-version checkpoint ABI.

### Documentation

- organized Qwen3.5-27B and Qwen3.5-35B-A3B performance evidence into matching
  layer-level and end-to-end serving sections;
- added reviewed TP2 serving manifests for both models and clarified the
  boundary between measured internal integration and supported public runtime
  integration;
- documented the publication rules for environment-bound integration snapshots.

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
