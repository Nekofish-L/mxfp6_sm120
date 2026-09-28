#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include "mxfp8_gemm/validation.hpp"

namespace mxfp8_sm120 {
__device__ __forceinline__ int scale_index(int row,int group,int k) {
  return (row/128)*(k/128)*512+(group/4)*512+(row%32)*16+((row%128)/32)*4+group%4;
}
__device__ __forceinline__ float scale_value(uint8_t x) {
  return x==255 ? __int_as_float(0x7fffffff) : __int_as_float(x==0 ? 0x00400000 : int(x)<<23);
}
template<int Rows>
__global__ void gemv(const uint32_t* __restrict__ a,const uint32_t* __restrict__ b,
                     const uint8_t* __restrict__ sa,const uint8_t* __restrict__ sb,
                     __nv_bfloat16* __restrict__ out,int m,int n,int k) {
  int lane=threadIdx.x%32,row=blockIdx.x*4+threadIdx.x/32;
  int mr=blockIdx.y*Rows;
  if(row>=n) return;
  float acc[Rows]={};
  for(int kk=lane*4;kk<k;kk+=128) {
    __nv_fp8x4_e4m3 w;w.__x=b[row*(k/4)+kk/4];
    float4 wf=static_cast<float4>(w);
    float ws=scale_value(sb[scale_index(row,kk/32,k)]);
#pragma unroll
    for(int i=0;i<Rows;++i) {
      if(mr+i<m) {
        __nv_fp8x4_e4m3 x;x.__x=a[(mr+i)*(k/4)+kk/4];
        float4 xf=static_cast<float4>(x);
        float xs=scale_value(sa[scale_index(mr+i,kk/32,k)]);
        float dot=xf.x*wf.x+xf.y*wf.y+xf.z*wf.z+xf.w*wf.w;
        acc[i]=fmaf(dot,xs*ws,acc[i]);
      }
    }
  }
#pragma unroll
  for(int i=0;i<Rows;++i) {
    for(int delta=16;delta>0;delta/=2) acc[i]+=__shfl_down_sync(0xffffffff,acc[i],delta);
    if(lane==0 && mr+i<m) out[(mr+i)*n+row]=__float2bfloat16_rn(acc[i]);
  }
}
void gemv_out(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,at::Tensor const& out,int64_t rows) {
  mxfp8_common::validate(a,b,sa,sb,out);
  c10::cuda::CUDAGuard guard(a.device());
  int m=a.size(0),n=b.size(0),k=a.size(1);
  TORCH_CHECK(rows==1 || rows==2 || rows==4 || rows==8,"Invalid GEMV row tile");
  dim3 grid((n+3)/4,(m+rows-1)/rows);
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
#define RUN(R) gemv<R><<<grid,128,0,stream>>>(reinterpret_cast<const uint32_t*>(a.data_ptr()),reinterpret_cast<const uint32_t*>(b.data_ptr()),reinterpret_cast<const uint8_t*>(sa.data_ptr()),reinterpret_cast<const uint8_t*>(sb.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),m,n,k)
  switch(rows) {case 1:RUN(1);break;case 2:RUN(2);break;case 4:RUN(4);break;case 8:RUN(8);break;}
#undef RUN
  auto error=cudaGetLastError();
  TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}
}
TORCH_LIBRARY_FRAGMENT(mxfp8_sm120,m) {m.def("gemv_out(Tensor a, Tensor b, Tensor sa, Tensor sb, Tensor(a!) out, int rows) -> ()");}
TORCH_LIBRARY_IMPL(mxfp8_sm120,CUDA,m) {m.impl("gemv_out",mxfp8_sm120::gemv_out);}
