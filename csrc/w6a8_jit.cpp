// Loaded lazily by prepare_w6a8(backend="jit"); no CUDA templates here.
#include <torch/library.h>
#include <deep_jit/backend/cuda/backend.hpp>
#include <c10/cuda/CUDAGuard.h>
#include "mxfp6_gemm/prepared_call.hpp"

namespace mxfp6_jit {
using Runtime = deep_jit::Runtime<deep_jit::CUDA>;
using Prepared = mxfp6_native::PreparedW6A8;
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuFuncGetParamInfo);

class Kernel : public torch::CustomClassHolder {
  std::shared_ptr<Runtime> runtime_;
  std::shared_ptr<deep_jit::cuda::Kernel> kernel_;
  std::string cubin_;
  int64_t config_id_, device_;
  at::ScalarType dtype_;
 public:
  Kernel(std::string const& root, std::string const& signature,
         std::string const& source, int64_t config_id, at::ScalarType dtype, int64_t device)
      : config_id_(config_id), device_(device), dtype_(dtype) {
    c10::cuda::CUDAGuard guard(device_);
    cudaStreamCaptureStatus status;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &status));
    TORCH_CHECK(status == cudaStreamCaptureStatusNone, "JIT compilation is forbidden during capture");
    auto* props = at::cuda::getDeviceProperties(device_);
    TORCH_CHECK(props->major == 12 && props->minor == 0, "W6A8 JIT requires SM120");
    TORCH_CHECK(config_id >= 0 && config_id < 30, "invalid W6A8 JIT config");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "invalid W6A8 JIT output dtype");
    runtime_ = std::make_shared<Runtime>(deep_jit::Config(
        root, "MXFP6", "w6a8-jit-v1-" + signature,
        {root + "/csrc/include", root + "/third_party/cutlass/include"}, {}));
    runtime_->default_compiler_options.arch = "120a";
    runtime_->default_compiler_options.ptxas_register_usage_level = 5;
    runtime_->default_compiler_options.extra_nvcc_flags = {
        "-DCUTLASS_ENABLE_GDC_FOR_SM100=1", "-DNDEBUG"};
    auto name = "w6a8_" + std::to_string(config_id) + "_" + std::to_string(static_cast<int>(dtype));
    auto path = runtime_->compile_without_load(name, source);
    kernel_ = runtime_->compile(name, source);
    cubin_ = (path / "kernel.cubin").string();
  }
  std::string cubin() const { return cubin_; }
  void bind(c10::intrusive_ptr<Prepared> const& prepared) {
    TORCH_CHECK(prepared->device_index() == device_ && prepared->config()[0] == config_id_ &&
                prepared->output_dtype() == dtype_, "JIT kernel and prepared configuration do not match");
    c10::cuda::CUDAGuard guard(device_);
    auto& launch = prepared->launch();
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "bind JIT before capture");
    // Binding an already-used object is cold. Finish prior launches before
    // replacing its host callable; existing graphs retain their own kernels
    // and the same owned parameter/workspace lifetime remains valid.
    if (launch.used) C10_CUDA_CHECK(cudaDeviceSynchronize());
    size_t offset = 0, bytes = 0;
    DJ_CUDA_DRIVER_CHECK(lazy_cuFuncGetParamInfo(kernel_->kernel_handle, 0, &offset, &bytes));
    TORCH_CHECK(offset == 0 && bytes == launch.parameter_bytes,
                "JIT CUTLASS Params ABI mismatch; rebuild against the matching source tree");
    auto runtime = runtime_;
    auto kernel = kernel_;
    auto parameters = launch.parameters;
    deep_jit::cuda::LaunchOptions options;
    options.grid_dim = launch.grid;
    options.block_dim = launch.block;
    options.num_smem_bytes = launch.shared_bytes;
    launch.run = [runtime, kernel, parameters, options](cudaStream_t stream, bool pdl) mutable {
      options.stream = stream;
      options.enable_pdl = pdl;
      runtime->launch(kernel, options, deep_jit::NoRefPtr{parameters.get()});
    };
  }
};

c10::intrusive_ptr<Kernel> compile(std::string const& root, std::string const& signature,
                                 std::string const& source, int64_t config_id,
                                 at::ScalarType dtype, int64_t device) {
  return c10::make_intrusive<Kernel>(root, signature, source, config_id, dtype, device);
}
} // namespace mxfp6_jit

TORCH_LIBRARY(mxfp6_jit, m) {
  m.class_<mxfp6_jit::Kernel>("Kernel")
      .def("bind", &mxfp6_jit::Kernel::bind)
      .def("cubin", &mxfp6_jit::Kernel::cubin);
  m.def("compile(str root, str signature, str source, int config_id, ScalarType dtype, int device) "
        "-> __torch__.torch.classes.mxfp6_jit.Kernel", &mxfp6_jit::compile);
}
