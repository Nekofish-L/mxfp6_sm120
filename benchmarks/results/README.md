# Autotuning results

`target_shapes.json` is the version-controlled output of the exhaustive
CUTLASS profiler search. It records the selected kernel for each of the 20
supported benchmark shapes and is retained as the reproducibility manifest for
the production dispatcher.

Raw profiler CSV files, repeated robustness runs, and experimental search
directories are generated artifacts and are intentionally ignored by Git. To
regenerate them, build the profiler and run `benchmarks/autotune.py` as
documented in the repository README.
