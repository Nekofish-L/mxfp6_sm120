#pragma once

#include <tuple>

#include <ATen/ATen.h>

namespace mxfp6_gemm::torch_ext {

// Dynamically quantize a contiguous FP16/BF16 [M,K] matrix with one
// power-of-two UE8M0 scale per 32 K values. Scales are emitted directly in the
// physical SM120 CUTLASS layout, including the required M padding.
std::tuple<at::Tensor, at::Tensor> quantize_mxfp8_cuda(
    at::Tensor const& input);
std::tuple<at::Tensor, at::Tensor> quantize_mxfp8_logical_cuda(
    at::Tensor const& input);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_mxfp8_dual_cuda(at::Tensor const& input);
void quantize_mxfp8_dual_out_cuda(
    at::Tensor& output,
    at::Tensor& logical_scales,
    at::Tensor& packed_scales,
    at::Tensor const& input);
std::tuple<at::Tensor, at::Tensor> quantize_mxfp6_cuda(
    at::Tensor const& input);

}  // namespace mxfp6_gemm::torch_ext
