// Shared by the fewwarp, oddwarp and oddrotate namespaces.
// No include guard: each family retains its original kernel and launch symbols.
template<int BM,int BN,int BK,int Stages,bool Swap,int FixedK=0,int WarpM=2,bool PackedSF=false,int Warps=4>
void launch(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,
            at::Tensor const& sb,at::Tensor const& out) {
  int m=Swap?b.size(0):a.size(0),n=Swap?a.size(0):b.size(0),k=a.size(1);
  constexpr int Bytes=(((BM+BN)*(BK+BK/32)+1023)/1024*1024)*Stages;
  auto fn=gemm<BM,BN,BK,Stages,Swap,FixedK,WarpM,PackedSF,Warps>;
  if constexpr(Bytes>=48*1024) {
    auto err=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,Bytes);
    TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
  }
  auto descriptor=[&](void* ptr,int rows,int tile_rows) {
    CUtensorMap map{};
    uint64_t dimensions[3]={128,uint64_t(rows),uint64_t(k/128)}, strides[2]={uint64_t(k),128};
    uint32_t box[3]={128,uint32_t(tile_rows),BK/128}, element_strides[3]={1,1,1};
    CUresult status=CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
      &map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,ptr,dimensions,strides,box,element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status==CUDA_SUCCESS,"TMA descriptor failed: ",int(status));
    return map;
  };
  auto ma=descriptor((Swap?b:a).data_ptr(),m,BM);
  auto mb=descriptor((Swap?a:b).data_ptr(),n,BN);
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
  bool pdl=mxfp8_runtime::pdl_enabled();
  auto err=mxfp8_runtime::launch_dependent_kernel(
    pdl,fn,dim3((n+BN-1)/BN,(m+BM-1)/BM),Warps*32,Bytes,stream,

    static_cast<const uint8_t*>((Swap?b:a).data_ptr()),
    static_cast<const uint8_t*>((Swap?a:b).data_ptr()),
    static_cast<const uint8_t*>((Swap?sb:sa).data_ptr()),
    static_cast<const uint8_t*>((Swap?sa:sb).data_ptr()),
    static_cast<__nv_bfloat16*>(out.data_ptr()),m,n,k,ma,mb,pdl);
  TORCH_CHECK(err==cudaSuccess,cudaGetErrorString(err));
}

template<int BM,int BN,int BK,int Stages,bool Swap,int WarpM=2,bool PackedSF=false,int Warps=4>
void specialized(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,at::Tensor const& out) {
  if(a.size(1)==2048) launch<BM,BN,BK,Stages,Swap,2048,WarpM,PackedSF,Warps>(a,b,sa,sb,out);
  else if(a.size(1)==2560) launch<BM,BN,BK,Stages,Swap,2560,WarpM,PackedSF,Warps>(a,b,sa,sb,out);
  else launch<BM,BN,BK,Stages,Swap,0,WarpM,PackedSF,Warps>(a,b,sa,sb,out);
}
