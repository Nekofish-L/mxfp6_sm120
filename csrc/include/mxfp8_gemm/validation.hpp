#pragma once
#include <climits>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
namespace mxfp8_common {
inline void validate(at::Tensor const& a,at::Tensor const& b,
                     at::Tensor const& sa,at::Tensor const& sb,at::Tensor const& out) {
  TORCH_CHECK(a.is_cuda() && a.dim()==2 && b.dim()==2,"Expected CUDA matrices");
  for(auto const& t:{a,b,sa,sb,out}) {
    TORCH_CHECK(t.device()==a.device() && t.is_contiguous(),"Expected contiguous tensors on one CUDA device");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%16==0,"Expected 16-byte-aligned storage");
  }
  int64_t m=a.size(0),n=b.size(0),k=a.size(1);
  TORCH_CHECK(a.scalar_type()==at::ScalarType::Float8_e4m3fn && b.scalar_type()==a.scalar_type(),"Expected E4M3 inputs");
  TORCH_CHECK(m>0 && n>0 && k>0 && m<=INT_MAX && n<=INT_MAX && k<=INT_MAX &&
              n%128==0 && k%128==0 && b.size(1)==k,"Expected positive M, N/K multiples of 128, matching K");
  TORCH_CHECK(sa.element_size()==1 && sb.element_size()==1 &&
              sa.numel()>=(m+127)/128*128*(k/32) && sb.numel()>=n*(k/32),
              "Expected padded 128x4 swizzled E8M0 scales");
  TORCH_CHECK(out.scalar_type()==at::kBFloat16 && out.dim()==2 &&
              out.size(0)==m && out.size(1)==n,"Expected BF16 output [M,N]");
  for(auto const& t:{a,b,sa,sb}) TORCH_CHECK(!out.is_alias_of(t),"Output must not alias inputs");
  auto* props=at::cuda::getDeviceProperties(a.get_device());
  TORCH_CHECK(props->major==12 && props->minor==0,"MXFP8 kernels require SM120");
}
inline void validate_scheduler(int splits,int swizzle) {
  TORCH_CHECK(splits>=1 && splits<=64 && (swizzle==1 || swizzle==2 || swizzle==4 || swizzle==8),
              "Invalid scheduler parameters");
}
}
