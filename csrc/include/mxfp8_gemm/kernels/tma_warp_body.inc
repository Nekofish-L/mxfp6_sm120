// Expanded inside each TMA kernel entry to preserve parameter and restrict scope.
// No include guard: used by multiple family entries.
// ReserveOutput requires the host to reserve max(pipeline bytes, output bytes).
  const int k=FixedK?FixedK:dynamic_k;
#if defined(__CUDA_ARCH_FEAT_SM120_ALL)
  extern __shared__ __align__(1024) uint8_t storage[];
  __shared__ uint64_t barriers[Stages];
  if(threadIdx.x==0) {
    for(int i=0;i<Stages;++i) cute::initialize_barrier(barriers[i]);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  constexpr int StageBytes=((BM+BN)*(BK+BK/32)+1023)/1024*1024;
  constexpr int TM=BM/(16*WarpM),TN=BN/(8*Warps/WarpM);
  int pm=blockIdx.y*BM,pn=blockIdx.x*BN;
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  int wm=(warp%WarpM)*16,wn=(warp/WarpM)*8;
  int row=lane/4,col=(lane%4)*4;
  float accum[TM][TN][4]={};
  mxfp8_runtime::dependent_prologue(pdl);
#pragma unroll
  for(int s=0;s<Stages-1;++s)
    prefetch<BM,BN,BK,Warps>(storage+s*StageBytes,a,b,sa,sb,pm,pn,s*BK,m,n,k,&ma,&mb,barriers+s);
  for(int start=0,it=0;start<k;start+=BK,++it) {
    int current=it%Stages;
    prefetch<BM,BN,BK,Warps>(storage+((it+Stages-1)%Stages)*StageBytes,
                      a,b,sa,sb,pm,pn,start+(Stages-1)*BK,m,n,k,&ma,&mb,barriers+((it+Stages-1)%Stages));
    asm volatile("cp.async.wait_group %0;" :: "n"(Stages-1));
    cute::wait_barrier(barriers[current],(it/Stages)&1);
    __syncthreads();
    const uint8_t* sm=storage+current*StageBytes;
    uint32_t as_pack[TM][BK/128],bs_pack[TN][BK/128];
    if constexpr(PackedSF) {
#pragma unroll
      for(int i=0;i<TM;++i) {
        int r=wm+i*(16*WarpM)+row+(lane%2)*8;
#pragma unroll
        for(int g=0;g<BK/128;++g)
          as_pack[i][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+BN)*BK+r*(BK/32)+g*4);
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        int r=BM+wn+j*(8*Warps/WarpM)+row;
#pragma unroll
        for(int g=0;g<BK/128;++g)
          bs_pack[j][g]=*reinterpret_cast<const uint32_t*>(sm+(BM+BN)*BK+r*(BK/32)+g*4);
      }
    }
#pragma unroll
    for(int kk=0;kk<BK;kk+=32) {
      uint32_t av[TM][4],bv[TN][2];
      uint8_t as[TM],bs[TN];
#pragma unroll
      for(int i=0;i<TM;++i) {
        int r=wm+i*(16*WarpM);
        const auto* ptr=reinterpret_cast<const cute::uint128_t*>(
            sm+sm_index<BK,BM>(r+lane%16,kk+(lane/16)*16));
        cute::SM75_U32x4_LDSM_N::copy(*ptr,av[i][0],av[i][1],av[i][2],av[i][3]);
        as[i]=PackedSF ? uint8_t(as_pack[i][kk/128] >> ((kk/32%4)*8)) : sm[(BM+BN)*BK+(r+row+(lane%2)*8)*(BK/32)+kk/32];
      }
#pragma unroll
      for(int j=0;j<TN;++j) {
        int r=wn+j*(8*Warps/WarpM)+row;
        const auto* ptr=reinterpret_cast<const cute::uint128_t*>(
            sm+BM*BK+sm_index<BK,BN>(wn+j*(8*Warps/WarpM)+lane%8,kk+((lane/8)%2)*16));
        cute::SM75_U32x2_LDSM_N::copy(*ptr,bv[j][0],bv[j][1]);
        bs[j]=PackedSF ? uint8_t(bs_pack[j][kk/128] >> ((kk/32%4)*8)) : sm[(BM+BN)*BK+(BM+r)*(BK/32)+kk/32];
      }
#pragma unroll
      for(int i=0;i<TM;++i) {
#pragma unroll
        for(int j=0;j<TN;++j) {
          auto& c=accum[i][j];
          Mma::fma(c[0],c[1],c[2],c[3],av[i][0],av[i][1],av[i][2],av[i][3],
                   bv[j][0],bv[j][1],c[0],c[1],c[2],c[3],as[i],bs[j]);
        }
      }
    }
    __syncthreads();
  }
  asm volatile("cp.async.wait_group 0;");
  // Drain every prefetched TMA before reusing its pipeline storage.
  for(int s=0;s<Stages;++s) {
    int last=(k+BK-1)/BK+Stages-2;
    if(s<=last) cute::wait_barrier(barriers[s],((last-s)/Stages)&1);
  }
  __syncthreads();
  constexpr int Rows=Swap?BN:BM, Cols=Swap?BM:BN, Stride=Cols+8;
  static_assert(ReserveOutput || Rows*Stride*2 <= StageBytes*Stages, "Output tile must fit the reused pipeline storage");
  auto* tile=reinterpret_cast<__nv_bfloat16*>(storage);
#pragma unroll
  for(int i=0;i<TM;++i) {
#pragma unroll
    for(int j=0;j<TN;++j) {
#pragma unroll
      for(int v=0;v<4;++v) {
        int r=wm+i*(16*WarpM)+row+(v/2)*8;
        int c=wn+j*(8*Warps/WarpM)+(lane%4)*2+v%2;
        tile[Swap?c*Stride+r:r*Stride+c]=__float2bfloat16_rn(accum[i][j][v]);
      }
    }
  }
  __syncthreads();
#pragma unroll
  for(int i=threadIdx.x;i<Rows*(Cols/8);i+=Warps*32) {
    int r=i/(Cols/8),c=(i%(Cols/8))*8;
    int gr=(Swap?pn:pm)+r,gc=(Swap?pm:pn)+c;
    int out_rows=Swap?n:m, out_cols=Swap?m:n;
    if(gr<out_rows && gc+7<out_cols) {
      *reinterpret_cast<uint4*>(out+int64_t(gr)*out_cols+gc)=
          *reinterpret_cast<const uint4*>(tile+r*Stride+c);
    }
  }
#else
  asm volatile("trap;");
#endif
