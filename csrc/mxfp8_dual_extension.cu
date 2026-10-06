// SPDX-License-Identifier: BSD-3-Clause
#include <ATen/ATen.h>
#include <torch/library.h>

#define DUAL_ARGS at::Tensor const& hi, at::Tensor const& weight, \
  at::Tensor const& s_hi, at::Tensor const& s_weight, \
  at::Tensor const& res, at::Tensor const& s_res, at::Tensor const& out
#define DUAL_CALL hi, weight, s_hi, s_weight, res, s_res, out

namespace mxfp8_dual_mlp32 { void gemm_out(DUAL_ARGS); }
namespace mxfp8_dual_qkvz32 { void gemm_out(DUAL_ARGS); }
namespace mxfp8_dual_qkvz64_pipereg { void gemm_out(DUAL_ARGS, int64_t config); }

namespace mxfp8_dual {
void gemm_out(DUAL_ARGS) {
  TORCH_CHECK(hi.dim()==2 && weight.dim()==2, "Expected hi and weight matrices");
  auto m=hi.size(0), n=weight.size(0), k=hi.size(1);
  TORCH_CHECK(k==2560 && weight.size(1)==k &&
              ((m==32 && (n==18432 || n==12288)) || (m==64 && n==12288)),
              "Unsupported dual MXFP8 shape; expected M32/N18432/K2560, "
              "M32/N12288/K2560 or M64/N12288/K2560");
  if (m==64) mxfp8_dual_qkvz64_pipereg::gemm_out(DUAL_CALL, 2);
  else if (n==18432) mxfp8_dual_mlp32::gemm_out(DUAL_CALL);
  else mxfp8_dual_qkvz32::gemm_out(DUAL_CALL);
}
}

TORCH_LIBRARY_FRAGMENT(mxfp8_sm120, m) {
  m.def("dual_gemm_out(Tensor hi, Tensor weight, Tensor s_hi, Tensor s_weight, "
        "Tensor res, Tensor s_res, Tensor(a!) out) -> ()");
}
TORCH_LIBRARY_IMPL(mxfp8_sm120, CUDA, m) {
  m.impl("dual_gemm_out", &mxfp8_dual::gemm_out);
}
