# Autotuning results

`target_shapes.json` records the final dispatch for the 30 Qwen3.5-27B TP2
benchmark shapes under the repository's `(M,N,K)` operator convention. Each
entry records compute type, orientation, tile, pipeline, scheduler, raster
order, and swizzle. The manifest distinguishes native W6A6 from mixed W6A8
paths and notes that mixed launches use programmatic dependent launch.
`general_policy.json` records the constraints, orientation boundary, portfolio,
and profiler-grid quality of the general native-W6A6 dispatcher.

Raw profiler CSV files, repeated robustness runs, and experimental search
directories are generated artifacts and are intentionally ignored by Git. The
profiler produces a top-K candidate set; final selection uses the complete
cold-cache PyTorch pipeline. Regenerate the search with
`benchmarks/autotune.py` as documented in the repository README.
