# Third-party notices

## NVIDIA CUTLASS

This repository pins NVIDIA CUTLASS at commit
`e6233cbac5d7c7a865c19c91cd684ceece19513c` as a Git submodule and carries
three source patches under `patches/cutlass/`.

CUTLASS is copyright NVIDIA Corporation and affiliates and is distributed
under the BSD 3-Clause License. A copy of that license is included in built
packages as `mxfp6/CUTLASS_LICENSE.txt`.

The checked-in CUTLASS patches are modifications maintained by this project;
they do not imply endorsement by NVIDIA.

## Humming benchmark reference

The reviewed benchmark metadata contains historical comparison measurements
against `inclusionAI/humming` at revision
`694298e9eb25ffdfc088353b49ba537ebf304479`. Humming source code is not copied,
vendored, linked or distributed by this project.
