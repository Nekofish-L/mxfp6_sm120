#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
// Accepted BA M16 prototype: kernel and staging constants are unchanged.
// Each CTA owns W[16,K] and X[8,K]. All K is staged once; four CTAs
// partition output coordinates, never the reduction dimension.
constexpr int K=2048, LD=K+8, SMEM=24*LD*2;
__global__ void ba_full(const __nv_bfloat16* __restrict__ x,
                        const __nv_bfloat16* __restrict__ w,
                        __nv_bfloat16* __restrict__ y) {
  extern __shared__ __align__(32) unsigned char storage[];
  auto *a=reinterpret_cast<__nv_bfloat16*>(storage);
  auto *b=a+16*LD;
  for(int v=threadIdx.x;v<24*K/8;v+=blockDim.x) {
    int r=v/(K/8), k=(v%(K/8))*8;
    const auto *src=r<16 ? w+(blockIdx.x*16+r)*K+k : x+(blockIdx.y*8+r-16)*K+k;
    auto *dst=r<16 ? a+r*LD+k : b+(r-16)*LD+k;
    *reinterpret_cast<int4*>(dst)=*reinterpret_cast<const int4*>(src);
  }
  __syncthreads();
  if(threadIdx.x<32) {
    int lane=threadIdx.x;
    float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
    #pragma unroll 8
    for(int k=0;k<K;k+=16) {
      uint32_t ar[4],br[2];
      uint32_t ap=static_cast<uint32_t>(__cvta_generic_to_shared(a+(lane%16)*LD+k+(lane/16)*8));
      uint32_t bp=static_cast<uint32_t>(__cvta_generic_to_shared(b+(lane%8)*LD+k+((lane/8)%2)*8));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(ar[0]),"=r"(ar[1]),"=r"(ar[2]),"=r"(ar[3]):"r"(ap));
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(br[0]),"=r"(br[1]):"r"(bp));
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(ar[0]),"r"(ar[1]),"r"(ar[2]),"r"(ar[3]),"r"(br[0]),"r"(br[1]));
    }
    int wr=blockIdx.x*16+lane/4, xr=blockIdx.y*8+(lane%4)*2;
    y[xr*32+wr]=__float2bfloat16_rn(c0);
    y[(xr+1)*32+wr]=__float2bfloat16_rn(c1);
    y[xr*32+wr+8]=__float2bfloat16_rn(c2);
    y[(xr+1)*32+wr+8]=__float2bfloat16_rn(c3);
  }
}
} // namespace

namespace mxfp8_bf16_ba {
at::Tensor bf16_ba_m16(at::Tensor const& x, at::Tensor const& weight) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "bf16_ba_m16 requires CUDA inputs");
  TORCH_CHECK(x.device() == weight.device(), "bf16_ba_m16 inputs must be on one CUDA device");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
              "bf16_ba_m16 requires BF16 inputs");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 16 && x.size(1) == K &&
              weight.dim() == 2 && weight.size(0) == 32 && weight.size(1) == K,
              "bf16_ba_m16 requires X[16,2048] and W[32,2048]");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
              "bf16_ba_m16 requires contiguous inputs");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
              "bf16_ba_m16 requires 16-byte-aligned inputs");
  c10::cuda::CUDAGuard guard(x.device());
  auto* props = at::cuda::getDeviceProperties(x.get_device());
  TORCH_CHECK(props->major == 12 && props->minor == 0, "bf16_ba_m16 requires SM120");
  auto output = at::empty({16, 32}, x.options());
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      ba_full, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  ba_full<<<dim3(2,2),512,SMEM,at::cuda::getCurrentCUDAStream(x.get_device())>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
} // namespace mxfp8_bf16_ba
