#pragma once
#include "mxfp_common/pdl.cuh"

namespace mxfp8_runtime {
bool pdl_enabled();
using mxfp_common::dependent_prologue;
using mxfp_common::launch_dependent_kernel;
} // namespace mxfp8_runtime
