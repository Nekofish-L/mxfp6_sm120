#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "mxfp6_gemm/kernel.hpp"

#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "helper.h"

namespace mx = mxfp6_gemm;
using namespace cute;

namespace {

struct Options {
  int m = 128;
  int n = 128;
  int k = 128;
  int iterations = 20;
  uint64_t seed = 2026;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool help = false;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    help = cmd.check_cmd_line_flag("help");
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
  }

  bool valid() const {
    return m > 0 && n > 0 && k > 0 &&
           n % 128 == 0 && k % 128 == 0 &&
           iterations >= 0;
  }

  double tflops(double runtime_ms) const {
    auto flops = 2.0 * static_cast<double>(m) * n * k;
    return flops / (runtime_ms * 1.0e9);
  }

  void usage(std::ostream& out, char const* executable) const {
    out << "Native SM120 MXFP6 E3M2 block-scaled GEMM\n\n"
        << "Usage: " << executable
        << " [--m=128] [--n=128] [--k=128] [--iterations=20]"
           " [--seed=2026] [--alpha=1] [--beta=0]\n\n"
        << "N and K must be positive multiples of 128; M must be positive. "
           "Inputs are packed"
           " FP6 E3M2 with one UE8M0 scale per 32 K elements.\n";
  }
};

using LayoutAStorage = decltype(cute::make_layout(
    cute::make_shape(0, 0, 0), mx::StrideA{}));
using LayoutBStorage = decltype(cute::make_layout(
    cute::make_shape(0, 0, 0), mx::StrideB{}));
using LayoutCStorage = decltype(cute::make_layout(
    cute::make_shape(0, 0, 0), mx::StrideC{}));
using LayoutDStorage = decltype(cute::make_layout(
    cute::make_shape(0, 0, 0), mx::StrideD{}));

struct Testbed {
  mx::StrideA stride_a;
  mx::StrideB stride_b;
  mx::StrideC stride_c;
  mx::StrideD stride_d;
  LayoutAStorage layout_a;
  LayoutBStorage layout_b;
  LayoutCStorage layout_c;
  LayoutDStorage layout_d;
  mx::LayoutSFA layout_sfa;
  mx::LayoutSFB layout_sfb;

  // HostTensor packs float_e3m2_t at six bits per logical element. Scale
  // tensors are allocated in the MMA-required interleaved layout below.
  cutlass::HostTensor<mx::ElementA, cutlass::layout::PackedVectorLayout> a;
  cutlass::HostTensor<mx::ElementB, cutlass::layout::PackedVectorLayout> b;
  cutlass::HostTensor<mx::ElementSF, cutlass::layout::PackedVectorLayout> sfa;
  cutlass::HostTensor<mx::ElementSF, cutlass::layout::PackedVectorLayout> sfb;
  cutlass::HostTensor<mx::ElementC, cutlass::layout::PackedVectorLayout> c;
  cutlass::HostTensor<mx::ElementD, cutlass::layout::PackedVectorLayout> d;
  cutlass::HostTensor<mx::ElementD, cutlass::layout::PackedVectorLayout> reference;

  template <class Element, class Layout>
  static void fill_random(cutlass::TensorView<Element, Layout> view,
                          uint64_t seed,
                          double maximum,
                          double minimum) {
    cutlass::reference::host::TensorFillRandomUniform(
        view, seed, maximum, minimum, 0);
  }

  void initialize(Options const& options) {
    auto problem = cute::make_shape(options.m, options.n, options.k, 1);
    stride_a = cutlass::make_cute_packed_stride(
        mx::StrideA{}, cute::make_shape(options.m, options.k, 1));
    stride_b = cutlass::make_cute_packed_stride(
        mx::StrideB{}, cute::make_shape(options.n, options.k, 1));
    stride_c = cutlass::make_cute_packed_stride(
        mx::StrideC{}, cute::make_shape(options.m, options.n, 1));
    stride_d = cutlass::make_cute_packed_stride(
        mx::StrideD{}, cute::make_shape(options.m, options.n, 1));

    layout_a = cute::make_layout(
        cute::make_shape(options.m, options.k, 1), stride_a);
    layout_b = cute::make_layout(
        cute::make_shape(options.n, options.k, 1), stride_b);
    layout_c = cute::make_layout(
        cute::make_shape(options.m, options.n, 1), stride_c);
    layout_d = cute::make_layout(
        cute::make_shape(options.m, options.n, 1), stride_d);
    layout_sfa = mx::BlockScaledConfig::tile_atom_to_shape_SFA(problem);
    layout_sfb = mx::BlockScaledConfig::tile_atom_to_shape_SFB(problem);

    a.reset(cutlass::make_Coord(cute::size(layout_a)));
    b.reset(cutlass::make_Coord(cute::size(layout_b)));
    sfa.reset(cutlass::make_Coord(cute::size(cute::filter_zeros(layout_sfa))));
    sfb.reset(cutlass::make_Coord(cute::size(cute::filter_zeros(layout_sfb))));
    c.reset(cutlass::make_Coord(cute::size(layout_c)));
    d.reset(cutlass::make_Coord(cute::size(layout_d)));
    reference.reset(cutlass::make_Coord(cute::size(layout_d)));

    // Values and scales are initialized independently. This exercises every
    // part of the MX representation without any software dequantization in the
    // device path. UE8M0 conversion rounds to a power of two.
    fill_random(a.host_view(), options.seed + 1, 2.0, -2.0);
    fill_random(b.host_view(), options.seed + 2, 2.0, -2.0);
    fill_random(sfa.host_view(), options.seed + 3, 4.0, 1.0);
    fill_random(sfb.host_view(), options.seed + 4, 4.0, 1.0);
    fill_random(c.host_view(), options.seed + 5, 2.0, -2.0);

    a.sync_device();
    b.sync_device();
    sfa.sync_device();
    sfb.sync_device();
    c.sync_device();
  }

  typename mx::Gemm::Arguments arguments(Options const& options) {
    return {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k, 1},
        {a.device_data(), stride_a,
         b.device_data(), stride_b,
         sfa.device_data(), layout_sfa,
         sfb.device_data(), layout_sfb},
        {{options.alpha, options.beta},
         c.device_data(), stride_c,
         d.device_data(), stride_d}};
  }

  template <class T>
  static auto iterator(T* ptr) {
    return cute::recast_ptr<T>(ptr);
  }

  bool verify(Options const& options) {
    auto tensor_a = cute::make_tensor(iterator(a.host_data()), layout_a);
    auto tensor_b = cute::make_tensor(iterator(b.host_data()), layout_b);
    auto tensor_sfa = cute::make_tensor(sfa.host_data(), layout_sfa);
    auto tensor_sfb = cute::make_tensor(sfb.host_data(), layout_sfb);
    auto tensor_c = cute::make_tensor(iterator(c.host_data()), layout_c);
    auto tensor_d = cute::make_tensor(iterator(reference.host_data()), layout_d);

    cutlass::reference::host::GettBlockScalingMainloopParams<
        mx::ElementAccumulator,
        decltype(tensor_a),
        decltype(tensor_sfa),
        decltype(tensor_b),
        decltype(tensor_sfb)>
        mainloop{tensor_a, tensor_sfa, tensor_b, tensor_sfb};

    cutlass::reference::host::GettBlockScalingEpilogueParams<
        mx::ElementCompute,
        mx::ElementAccumulator,
        mx::ElementCompute,
        decltype(tensor_c),
        decltype(tensor_d)>
        epilogue{options.alpha, options.beta, tensor_c, tensor_d};

    cutlass::reference::host::Gemm3x(mainloop, epilogue);
    d.sync_host();

    bool equal = cutlass::reference::host::TensorEquals(
        reference.host_view(), d.host_view());
    bool nonzero = cutlass::reference::host::TensorNorm(d.host_view()) > 0;
    return equal && nonzero;
  }
};

int run(Options const& options) {
  Testbed testbed;
  testbed.initialize(options);
  auto arguments = testbed.arguments(options);

  mx::Gemm gemm;
  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS can_implement failed: "
              << cutlassGetStatusString(status) << '\n';
    return EXIT_FAILURE;
  }

  cutlass::device_memory::allocation<uint8_t> workspace(
      mx::Gemm::get_workspace_size(arguments));
  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS initialize failed: "
              << cutlassGetStatusString(status) << '\n';
    return EXIT_FAILURE;
  }

  status = gemm.run();
  if (status != cutlass::Status::kSuccess || cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "CUTLASS kernel launch failed\n";
    return EXIT_FAILURE;
  }

  bool passed = testbed.verify(options);
  std::cout << "Verification: " << (passed ? "PASSED" : "FAILED") << '\n';
  if (!passed) {
    return EXIT_FAILURE;
  }

  if (options.iterations > 0) {
    GpuTimer timer;
    timer.start();
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      status = gemm.run();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS profiling launch failed\n";
        return EXIT_FAILURE;
      }
    }
    timer.stop();
    double runtime_ms = timer.elapsed_millis() / options.iterations;
    std::cout << std::fixed << std::setprecision(4)
              << "Runtime: " << runtime_ms << " ms\n"
              << "Throughput: " << options.tflops(runtime_ms) << " TFLOP/s\n";
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char const** argv) {
#if !defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  std::cerr << "This binary must be compiled for sm_120a.\n";
  return EXIT_FAILURE;
#else
  Options options;
  options.parse(argc, argv);
  if (options.help) {
    options.usage(std::cout, argv[0]);
    return EXIT_SUCCESS;
  }
  if (!options.valid()) {
    options.usage(std::cerr, argv[0]);
    std::cerr << "\nInvalid problem shape or iteration count.\n";
    return EXIT_FAILURE;
  }

  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    std::cerr << "Unable to query the CUDA device.\n";
    return EXIT_FAILURE;
  }
  if (properties.major != 12 || properties.minor != 0) {
    std::cerr << "SM120 is required; found compute capability "
              << properties.major << '.' << properties.minor << ".\n";
    return EXIT_FAILURE;
  }

  std::cout << "Device: " << properties.name << " (SM120)\n"
            << "Kernel: mma.sync.aligned.kind::mxf8f6f4.block_scale, "
               "E3M2 x E3M2, UE8M0/32, FP32 accumulate, FP16 output\n"
            << "Problem: " << options.m << 'x' << options.n << 'x'
            << options.k << '\n';
  return run(options);
#endif
}
