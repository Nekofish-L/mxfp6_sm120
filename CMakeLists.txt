cmake_minimum_required(VERSION 3.24)

project(mxfp6_sm120_cutlass LANGUAGES CXX CUDA)

option(MXFP6_BUILD_STANDALONE "Build the standalone benchmark executable" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass" CACHE PATH
    "Path to a CUTLASS checkout with SM120 MXF8F6F4 support")

if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR
    "CUTLASS was not found at ${CUTLASS_DIR}. Run: git submodule update "
    "--init --depth 1 third_party/cutlass")
endif()

file(READ
  "${CUTLASS_DIR}/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl"
  MXFP6_CUTLASS_SM120_BLOCKSCALED_BUILDER)
file(READ
  "${CUTLASS_DIR}/include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl"
  MXFP6_CUTLASS_SM120_BLOCKWISE_BUILDER)
if(NOT MXFP6_CUTLASS_SM120_BLOCKSCALED_BUILDER MATCHES "sSFATileShape_M" OR
   NOT MXFP6_CUTLASS_SM120_BLOCKWISE_BUILDER MATCHES "SmemCopyOpB")
  message(FATAL_ERROR
    "The required SM120 small-tile CUTLASS patch is not applied. Run: "
    "bash scripts/apply_cutlass_patches.sh --runtime-only")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
# Torch's package appends a broad set of -gencode flags globally. This project
# deliberately owns CUDA code generation because block-scaled MMA needs 120a.
set(MXFP6_SAVED_CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
set(MXFP6_SAVED_CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
find_package(Torch CONFIG REQUIRED)
set(CUDA_NVCC_FLAGS "${MXFP6_SAVED_CUDA_NVCC_FLAGS}")
set(CMAKE_CUDA_FLAGS "${MXFP6_SAVED_CMAKE_CUDA_FLAGS}")

if(MXFP6_BUILD_STANDALONE)
  add_executable(mxfp6_gemm
    csrc/mxfp6_gemm.cu
  )

  target_include_directories(mxfp6_gemm PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include"
    "${CUTLASS_DIR}/include"
    "${CUTLASS_DIR}/tools/util/include"
    "${CUTLASS_DIR}/examples/common"
  )

  # CMake's numeric CUDA_ARCHITECTURES property cannot express the architecture-
  # accelerated `a` target. Block-scaled MMA requires sm_120a, not sm_120f.
  target_compile_options(mxfp6_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      -arch=sm_120a
      --expt-relaxed-constexpr
      -O3
      -Xcompiler=-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.
      -Xcompiler=-fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.
      -Xcudafe=--diag_suppress=20012
      -Xcudafe=--diag_suppress=20013
      -Xcudafe=--diag_suppress=20015
    >
  )

  set_target_properties(mxfp6_gemm PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_SEPARABLE_COMPILATION OFF
  )
endif()

# A dispatcher-only shared library: Python loads it with
# torch.ops.load_library(), so no CPython/PYBIND11 module is required.
add_library(mxfp6_torch SHARED
  csrc/torch_extension.cu
  csrc/packing.cu
  csrc/quantization.cu
)

target_include_directories(mxfp6_torch PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include"
  "${CUTLASS_DIR}/include"
  "${CUTLASS_DIR}/tools/util/include"
)

target_link_libraries(mxfp6_torch PRIVATE ${TORCH_LIBRARIES})
target_compile_options(mxfp6_torch PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
      -arch=sm_120a
      -DCUTLASS_ENABLE_GDC_FOR_SM100=1
      --expt-relaxed-constexpr
      -O3
      -Xcompiler=-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.
      -Xcompiler=-fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.
      -Xcudafe=--diag_suppress=20012
      -Xcudafe=--diag_suppress=20013
      -Xcudafe=--diag_suppress=20015
  >
)

set_target_properties(mxfp6_torch PROPERTIES
  CUDA_ARCHITECTURES OFF
  CUDA_SEPARABLE_COMPILATION OFF
  PREFIX ""
  OUTPUT_NAME "mxfp6_torch"
  # The extension is loaded after `import torch`, so libtorch's DSOs are
  # already resident. Never embed the build environment's absolute torch/lib
  # path in a distributable wheel.
  BUILD_WITH_INSTALL_RPATH ON
  INSTALL_RPATH "$ORIGIN"
)

install(TARGETS mxfp6_torch
  LIBRARY DESTINATION mxfp6
)

include(CTest)
if(BUILD_TESTING AND MXFP6_BUILD_STANDALONE)
  add_test(NAME mxfp6_gemm_128
    COMMAND mxfp6_gemm --m=128 --n=128 --k=128 --iterations=0)
  add_test(NAME mxfp6_gemm_m32
    COMMAND mxfp6_gemm --m=32 --n=128 --k=128 --iterations=0)
  add_test(NAME mxfp6_torch_python
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/benchmark.py"
            --library=$<TARGET_FILE:mxfp6_torch>
            --shapes=1x8x128,17x136x128,96x136x128,112x136x128,129x128x128
            --warmup=1 --iterations=2
            --flush-l2-mb=0 --no-compare-fp8 --check-all)
  add_test(NAME mxfp6_torch_tools
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ops.py"
            --library=$<TARGET_FILE:mxfp6_torch>)
endif()
