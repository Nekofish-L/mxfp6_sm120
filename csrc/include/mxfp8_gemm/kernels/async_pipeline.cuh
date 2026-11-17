#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "mxfp8_gemm/layout.cuh"

namespace mxfp8_pipeline {
using mxfp8_layout::sf_index;
using mxfp8_layout::row_major::sm_index;

template<bool CacheAll>
__device__ __forceinline__ void copy16(uint8_t* dst,const uint8_t* src,bool valid) {
  uint32_t addr=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  if constexpr (CacheAll) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::
                 "r"(addr),"l"(src),"r"(valid?16:0));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                 "r"(addr),"l"(src),"r"(valid?16:0));
  }
}
__device__ __forceinline__ void copy4(uint8_t* dst,const uint8_t* src,bool valid) {
  uint32_t addr=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
               "r"(addr),"l"(src),"r"(valid?4:0));
}
template<int BM,int BN,int BK,int Threads=128,bool CacheAll=false>
__device__ void prefetch(uint8_t* sm,const uint8_t* a,const uint8_t* b,
                         const uint8_t* sa,const uint8_t* sb,
                         int pm,int pn,int start,int m,int n,int k) {
#pragma unroll
  for(int i=threadIdx.x;i<(BM+BN)*(BK/16);i+=Threads) {
    int row=i/(BK/16),col=(i%(BK/16))*16;
    bool is_a=row<BM;
    int r=is_a?pm+row:pn+row-BM;
    int limit=is_a?m:n;
    bool valid=r<limit && start+col<k;
    const uint8_t* src=(is_a?a:b)+(valid?r*k+start+col:0);
    copy16<CacheAll>(sm+sm_index<BK>(row,col),src,valid);
  }
#pragma unroll
  for(int i=threadIdx.x;i<(BM+BN)*(BK/128);i+=Threads) {
    int row=i/(BK/128),g=(i%(BK/128))*4;
    bool is_a=row<BM;
    int r=is_a?pm+row:pn+row-BM;
    bool valid=r<(is_a?m:n) && start+g*32<k;
    const uint8_t* src=(is_a?sa:sb)+(valid?sf_index(r,start/32+g,k):0);
    copy4(sm+(BM+BN)*BK+row*(BK/32)+g,src,valid);
  }
  asm volatile("cp.async.commit_group;");
}
} // namespace mxfp8_pipeline
