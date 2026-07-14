# Autotuning results

`target_shapes.json` records exact cold-cache overrides for the original 20
benchmark shapes. `general_policy.json` records the constraints, orientation
boundary, portfolio, and profiler-grid quality of the general dispatcher.

Raw profiler CSV files, repeated robustness runs, and experimental search
directories are generated artifacts and are intentionally ignored by Git. To
regenerate them, build the profiler and run `benchmarks/autotune.py` as
documented in the repository README.
