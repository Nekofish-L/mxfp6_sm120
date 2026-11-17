#pragma once

// SM120 E8M0 scale layout: 128 rows x four scale groups per atom.
// Keep row-major cp.async and 128-byte TMA-box layouts distinct. BK is
// retained in the TMA signature to preserve existing kernel call sites.
namespace mxfp8_layout {
__device__ __forceinline__ int sf_index(int r,int g,int k) {
  return (r/128)*(k/128)*512+(g/4)*512+(r%32)*16+((r%128)/32)*4+g%4;
}
namespace row_major {
template<int BK>
__device__ __forceinline__ int sm_index(int r,int c) {
  return r*BK+(c^((r%8)*16));
}
}
namespace tma {
template<int BK,int Rows>
__device__ __forceinline__ int sm_index(int r,int c) {
  return (c/128)*Rows*128+r*128+((c%128)^((r%8)*16));
}
}
} // namespace mxfp8_layout
