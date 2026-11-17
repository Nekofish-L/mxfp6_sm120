"""Check shared address helpers against the independent packed layout order."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_layout_helpers(tmp_path):
    compiler=shutil.which('c++')
    if compiler is None:
        pytest.skip('host C++ compiler required')
    source=tmp_path/'layout.cpp'
    source.write_text('''
#define __device__
#define __forceinline__ inline
#include "mxfp8_gemm/layout.cuh"
extern "C" int sf(int r,int g,int k) {return mxfp8_layout::sf_index(r,g,k);}
extern "C" int row(int r,int c) {return mxfp8_layout::row_major::sm_index<256>(r,c);}
extern "C" int tma(int r,int c) {return mxfp8_layout::tma::sm_index<256,64>(r,c);}
''')
    library=tmp_path/'layout.so'
    subprocess.run([compiler,'-std=c++17','-shared','-fPIC','-O2','-I',str(ROOT/'csrc/include'),str(source),'-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library))
    for k in (128,256,2560):
        index=0
        for atom in range(2):
            for kb in range(k//128):
                for lane in range(32):
                    for quad in range(4):
                        for group in range(4):
                            assert lib.sf(atom*128+quad*32+lane,kb*4+group,k)==index
                            index+=1
    for r in range(64):
        for c in range(256):
            swizzled=((c//16)^(r%8))*16+c%16
            assert lib.row(r,c)==r*256+swizzled
            assert lib.tma(r,c)==(c//128)*64*128+r*128+swizzled%128
