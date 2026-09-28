"""Independent-CTA SM120 block-scaled MMA for wide-N, short-K GEMMs."""
import torch
import triton
import triton.language as tl


CONFIGS = (
    (64,64,256,4,3,1,True), (64,64,256,4,3,1,False),
    (16,128,256,4,2,1,False), (128,16,128,4,3,1,True),
    (64,128,256,4,3,1,False), (128,128,128,4,3,1,False),
)


@triton.jit
def _sf_offset(r, g, K: tl.constexpr):
    return (r//128)*(K//128)*512+(g//4)*512+(r%32)*16+((r%128)//32)*4+g%4


@triton.jit
def _mm(A,B,SA,SB,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GROUP:tl.constexpr,
        TRANSPOSE_OUTPUT:tl.constexpr=False):
    pid=tl.program_id(0)
    nm=tl.cdiv(M,BM);nn=tl.cdiv(N,BN)
    group=pid//(GROUP*nn)
    first=group*GROUP
    gs=tl.minimum(nm-first,GROUP)
    pm=first+pid%gs
    pn=(pid%(GROUP*nn))//gs
    rm=pm*BM+tl.arange(0,BM)
    rn=pn*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    rg=tl.arange(0,BK//32)
    acc=tl.full((BM,BN),0,tl.float32)
    for start in range(tl.cdiv(K,BK)):
        ki=start*BK+rk
        gi=start*(BK//32)+rg
        a=tl.load(A+rm[:,None]*K+ki[None,:],(rm[:,None]<M)&(ki[None,:]<K),0)
        b=tl.load(B+rn[None,:]*K+ki[:,None],(rn[None,:]<N)&(ki[:,None]<K),0)
        sa=tl.load(SA+_sf_offset(rm[:,None],gi[None,:],K),(rm[:,None]<M)&(gi[None,:]<K//32),127)
        sb=tl.load(SB+_sf_offset(rn[:,None],gi[None,:],K),(rn[:,None]<N)&(gi[None,:]<K//32),127)
        acc=tl.dot_scaled(a,sa,'e4m3',b,sb,'e4m3',acc)
    if TRANSPOSE_OUTPUT:
        dst=O+rn[None,:]*M+rm[:,None]
    else:
        dst=O+rm[:,None]*N+rn[None,:]
    tl.store(dst,acc,(rm[:,None]<M)&(rn[None,:]<N))


def run(a,b,sa,sb,out,config):
    bm,bn,bk,warps,stages,group=config[:6]
    swapped=bool(config[6]) if len(config)>6 else False
    if swapped:
        a,b,sa,sb=b,a,sb,sa
    m,k=a.shape;n=b.shape[0]
    return _mm[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](a.view(torch.uint8),b.view(torch.uint8),sa.view(torch.uint8).view(-1),sb.view(torch.uint8).view(-1),out,m,n,k,bm,bn,bk,group,swapped,num_warps=warps,num_stages=stages)
