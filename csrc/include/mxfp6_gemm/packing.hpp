#pragma once

#include <ATen/ATen.h>

namespace mxfp6_gemm::torch_ext {

// Pack four logical E3M2 bit patterns into three bytes. Only the low six bits
// of every input byte are significant.
at::Tensor pack_fp6_cuda(at::Tensor const& codes);

// Inverse of pack_fp6_cuda. The returned tensor has shape [rows, k].
at::Tensor unpack_fp6_cuda(at::Tensor const& packed, int64_t rows, int64_t k);

// Convert logical UE8M0 scales [rows, k / 32] to and from the physical SM120
// block-scale layout used by CUTLASS. Packed output is a flat byte tensor and
// includes the required padding to a multiple of 128 rows.
at::Tensor pack_scales_cuda(
    at::Tensor const& logical, int64_t rows, int64_t k);
at::Tensor unpack_scales_cuda(
    at::Tensor const& packed, int64_t rows, int64_t k);

}  // namespace mxfp6_gemm::torch_ext
