// SPDX-License-Identifier: BSD-3-Clause
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/library.h>

#define DUAL_ARGS at::Tensor const& hi, at::Tensor const& weight, \
  at::Tensor const& s_hi, at::Tensor const& s_weight, \
  at::Tensor const& res, at::Tensor const& s_res, at::Tensor const& out, bool use_pdl
#define DUAL_CALL hi, weight, s_hi, s_weight, res, s_res, out, use_pdl

namespace mxfp8_dual_small { void gemm_out(DUAL_ARGS); }
namespace mxfp8_dual_pipereg { void gemm_out(DUAL_ARGS); }

namespace mxfp8_dual {
void gemm_out(DUAL_ARGS) {
  TORCH_CHECK(hi.is_cuda() && hi.dim()==2 && weight.dim()==2,
              "Expected CUDA hi and weight matrices");
  auto m=hi.size(0), n=weight.size(0), k=hi.size(1);
  TORCH_CHECK(k==2560 && weight.size(1)==k &&
              m>=1 && m<=128 && (n==18432 || n==12288),
              "Unsupported dual MXFP8 shape; expected 1<=M<=128, "
              "N18432 or N12288, K2560");
  TORCH_CHECK(res.dim()==2 && res.size(0)==m && res.size(1)==k,
              "Expected res [M,2560]");
  TORCH_CHECK(hi.scalar_type()==at::ScalarType::Float8_e4m3fn &&
              res.scalar_type()==hi.scalar_type() && weight.scalar_type()==hi.scalar_type(),
              "Expected E4M3FN values");
  TORCH_CHECK(s_hi.scalar_type()==at::kByte && s_res.scalar_type()==at::kByte &&
              s_weight.scalar_type()==at::kByte && s_hi.dim()==1 &&
              s_res.dim()==1 && s_weight.dim()==1 && s_hi.numel()==128*(k/32) &&
              s_res.numel()==128*(k/32) && s_weight.numel()==n*(k/32),
              "Expected exact packed E8M0 scale storage");
  TORCH_CHECK(out.scalar_type()==at::kBFloat16 && out.dim()==2 &&
              out.size(0)==m && out.size(1)==n,"Expected BF16 out [M,N]");
  for(auto const& t:{hi,weight,s_hi,s_weight,res,s_res,out}) {
    TORCH_CHECK(t.device()==hi.device() && t.is_contiguous(),
                "Expected contiguous tensors on one CUDA device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,
                "Expected 16-byte-aligned storage");
  }
  for(auto const& t:{hi,weight,s_hi,s_weight,res,s_res})
    TORCH_CHECK(!out.is_alias_of(t),"Output must not alias inputs");
  c10::cuda::CUDAGuard guard(hi.device());
  cudaDeviceProp props{};
  auto err=cudaGetDeviceProperties(&props,hi.get_device());
  TORCH_CHECK(err==cudaSuccess,"cudaGetDeviceProperties failed: ",cudaGetErrorString(err));
  TORCH_CHECK(props.major==12 && props.minor==0,"Requires SM120a");
  if (m>32) mxfp8_dual_pipereg::gemm_out(DUAL_CALL);
  else mxfp8_dual_small::gemm_out(DUAL_CALL);
}
}

TORCH_LIBRARY_FRAGMENT(mxfp8_sm120, m) {
  m.def("dual_gemm_out(Tensor hi, Tensor weight, Tensor s_hi, Tensor s_weight, "
        "Tensor res, Tensor s_res, Tensor(a!) out, bool use_pdl=False) -> ()");
}
TORCH_LIBRARY_IMPL(mxfp8_sm120, CUDA, m) {
  m.impl("dual_gemm_out", &mxfp8_dual::gemm_out);
}
