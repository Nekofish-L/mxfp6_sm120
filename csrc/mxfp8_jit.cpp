// Optional host bridge, built only when the explicit JIT API is first used.
#include <torch/extension.h>
#include <deep_jit/backend/cuda/backend.hpp>
#include "mxfp8_gemm/validation.hpp"
#include <c10/cuda/CUDAGuard.h>

namespace {
using Runtime = deep_jit::Runtime<deep_jit::CUDA>;
struct CompiledKernel {
  std::shared_ptr<Runtime> runtime;
  std::shared_ptr<deep_jit::cuda::Kernel> kernel;
  int device, bm, bn, threads, shared_bytes, fixed_k;
  bool swap;
  std::string cubin;
};

std::shared_ptr<CompiledKernel> compile_kernel(
    const std::string& root, const std::string& signature, const std::string& source,
    const std::string& name, int device, int bm, int bn, int threads,
    int shared_bytes, int fixed_k, bool swap) {
  c10::cuda::CUDAGuard guard(device);
  cudaStreamCaptureStatus status;
  TORCH_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &status) == cudaSuccess &&
              status == cudaStreamCaptureStatusNone, "JIT compilation is forbidden during capture");
  auto* props = at::cuda::getDeviceProperties(device);
  TORCH_CHECK(props->major == 12 && props->minor == 0, "MXFP8 JIT requires SM120");
  TORCH_CHECK(bm > 0 && bn > 0 && (threads == 128 || threads == 256) && shared_bytes > 0 &&
              (fixed_k == 0 || fixed_k == 2048 || fixed_k == 2560), "Invalid JIT descriptor");
  auto runtime = std::make_shared<Runtime>(deep_jit::Config(
      root, "MXFP8", "mxfp8-jit-v1-" + signature,
      {root + "/csrc/include", root + "/third_party/cutlass/include"}, {}));
  runtime->default_compiler_options.arch = "120a";
  runtime->default_compiler_options.ptxas_register_usage_level = 5;
  runtime->default_compiler_options.extra_nvcc_flags.push_back("-DCUTLASS_ENABLE_GDC_FOR_SM100=1");
  auto path = runtime->compile_without_load(name, source);
  auto kernel = runtime->compile(name, source);
  return std::make_shared<CompiledKernel>(CompiledKernel{
      runtime, kernel, device, bm, bn, threads, shared_bytes, fixed_k, swap,
      (path / "kernel.cubin").string()});
}

class PreparedCall {
  std::shared_ptr<CompiledKernel> compiled;
  at::Tensor a, b, sa, sb, out;
  int m, n, k;
  bool pdl;
 public:
  PreparedCall(std::shared_ptr<CompiledKernel> compiled, at::Tensor a, at::Tensor b,
               at::Tensor sa, at::Tensor sb, at::Tensor out, bool pdl)
      : compiled(std::move(compiled)), a(a), b(b), sa(sa), sb(sb), out(out), pdl(pdl) {
    mxfp8_common::validate(a, b, sa, sb, out);
    TORCH_CHECK(a.get_device() == this->compiled->device, "JIT handle belongs to another device");
    m = a.size(0); n = b.size(0); k = a.size(1);
    TORCH_CHECK(this->compiled->fixed_k == 0 || this->compiled->fixed_k == k,
                "JIT handle has a different fixed K");
  }

  at::Tensor run() {
    c10::cuda::CUDAGuard guard(compiled->device);
    mxfp8_common::validate(a, b, sa, sb, out);
    TORCH_CHECK(a.size(0) == m && b.size(0) == n && a.size(1) == k,
                "Prepared JIT operands must not be resized");
    int rows = compiled->swap ? n : m, cols = compiled->swap ? m : n;
    deep_jit::cuda::LaunchOptions options;
    options.num_smem_bytes = compiled->shared_bytes;
    options.grid_dim = dim3((cols + compiled->bn - 1) / compiled->bn,
                           (rows + compiled->bm - 1) / compiled->bm);
    options.block_dim = dim3(compiled->threads);
    options.enable_pdl = pdl;
    compiled->runtime->launch(compiled->kernel, options,
        (compiled->swap ? b : a).data_ptr(), (compiled->swap ? a : b).data_ptr(),
        (compiled->swap ? sb : sa).data_ptr(), (compiled->swap ? sa : sb).data_ptr(),
        out.data_ptr(), rows, cols, k, pdl);
    return out;
  }
};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<CompiledKernel, std::shared_ptr<CompiledKernel>>(module, "CompiledKernel")
      .def_readonly("cubin", &CompiledKernel::cubin)
      .def_readonly("device", &CompiledKernel::device);
  pybind11::class_<PreparedCall>(module, "PreparedCall")
      .def(pybind11::init<std::shared_ptr<CompiledKernel>, at::Tensor, at::Tensor,
                         at::Tensor, at::Tensor, at::Tensor, bool>())
      .def("__call__", &PreparedCall::run);
  module.def("compile_kernel", &compile_kernel);
}
