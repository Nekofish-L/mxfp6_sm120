#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <optional>
#include <vector>
#include "mxfp8_gemm/validation.hpp"
#include "mxfp8_gemm/workspace.hpp"

#define MXFP8_ARGS at::Tensor const& a, at::Tensor const& b, at::Tensor const& sa, at::Tensor const& sb, at::Tensor const& out
#define MXFP8_NATIVE(NAME) namespace NAME { int64_t mm_out(MXFP8_ARGS, int64_t tactic, int64_t splits, int64_t swizzle, std::optional<at::Tensor> workspace, bool query, int64_t sms); }
MXFP8_NATIVE(mxfp8_sm120)
MXFP8_NATIVE(mxfp8_direct)
MXFP8_NATIVE(mxfp8_dynamic)
MXFP8_NATIVE(mxfp8_extra)
MXFP8_NATIVE(mxfp8_wide)
MXFP8_NATIVE(mxfp8_release_core)
MXFP8_NATIVE(mxfp8_release_direct)
#undef MXFP8_NATIVE
namespace mxfp8_sm120 { void gemv_out(MXFP8_ARGS, int64_t rows); }
#define MXFP8_FAMILY(NAME, BASE, COUNT, RETURN) namespace mxfp8_##NAME { RETURN mm_out(MXFP8_ARGS, int64_t config); }
#include "mxfp8_gemm/registry.def"
#undef MXFP8_FAMILY

namespace mxfp8_runtime {
struct LaunchConfig { int64_t tactic, splits, swizzle, sms; };
WorkspacePool& workspace_pool() { static WorkspacePool pool; return pool; }
}
#include "mxfp8_gemm/dispatch_policy.hpp"

namespace mxfp8_runtime {
bool core_tactic(int64_t t) { return t >= 0 && t < 96 && t != 30 && !(t >= 80 && t < 86); }
bool release_direct_base(int64_t t) {
  return (t >= 43 && t <= 50) || (t >= 58 && t <= 69) || (t >= 86 && t <= 95);
}
bool valid_tactic(int64_t t) {
  if (core_tactic(t)) return true;
  if (t >= 300 && t < 500) return core_tactic(t % 100) && !(t % 100 >= 33 && t % 100 <= 36);
  if (t >= 800 && t < 1100) return t % 100 < 33 && t % 100 != 30;
  if (t >= 1300 && t < 1600) return release_direct_base(t % 100);
#define MXFP8_FAMILY(NAME, BASE, COUNT, RETURN) if (t >= BASE && t < BASE + COUNT) return true;
#include "mxfp8_gemm/registry.def"
#undef MXFP8_FAMILY
  return false;
}
bool stream_k_tactic(int64_t t) {
  if ((t >= 300 && t < 500) || (t >= 800 && t < 1100)) t %= 100;
  return (t >= 12 && t <= 16) || t == 39 || t == 40 || t == 42 || t == 51 || t == 57;
}
std::vector<int64_t> tactics(bool stream_k_only) {
  std::vector<int64_t> result;
  for (int64_t t = 0; t < 1600; ++t)
    if (valid_tactic(t) && (!stream_k_only || stream_k_tactic(t))) result.push_back(t);
  return result;
}
std::vector<int64_t> config_values(LaunchConfig c) { return {c.tactic,c.splits,c.swizzle,c.sms}; }
std::vector<int64_t> selected_config(int64_t m,int64_t n,int64_t k) { return config_values(select_config(m,n,k)); }
std::string policy_hash() { return policy_sha256; }

LaunchConfig resolve(at::Tensor const& a, at::Tensor const& b, int64_t tactic, int64_t splits, int64_t swizzle, int64_t sms) {
  TORCH_CHECK(a.dim()==2 && b.dim()==2, "Expected CUDA matrices");
  auto config = tactic == -1 ? select_config(a.size(0),b.size(0),a.size(1)) : LaunchConfig{tactic,splits,swizzle,sms};
  TORCH_CHECK(valid_tactic(config.tactic), "Unknown MXFP8 tactic: ",config.tactic);
  return config;
}

int64_t dispatch_out(MXFP8_ARGS, LaunchConfig c, std::optional<at::Tensor> workspace, bool query) {
  mxfp8_common::validate(a,b,sa,sb,out);
  mxfp8_common::validate_scheduler(c.splits,c.swizzle);
  auto t=c.tactic;
#define CALL(NS) return NS::mm_out(a,b,sa,sb,out,t,c.splits,c.swizzle,workspace,query,c.sms)
  if (t >= 1300 && t < 1600) { CALL(mxfp8_release_direct); }
  if (t >= 800 && t < 1100) { CALL(mxfp8_release_core); }
#define MXFP8_FAMILY(NAME, BASE, COUNT, RETURN) \
  if (t >= BASE && t < BASE + COUNT) { if (!query) mxfp8_##NAME::mm_out(a,b,sa,sb,out,t-BASE); return 0; }
#include "mxfp8_gemm/registry.def"
#undef MXFP8_FAMILY
  auto base = t >= 300 && t < 500 ? t % 100 : t;
  if (base >= 33 && base <= 36) {
    if (!query) mxfp8_sm120::gemv_out(a,b,sa,sb,out,int64_t(1) << (base-33));
    return 0;
  }
  if (base >= 70 && base <= 79) { CALL(mxfp8_dynamic); }
  if (base >= 51 && base <= 57) { CALL(mxfp8_extra); }
  if (base >= 43) { CALL(mxfp8_direct); }
  if (base >= 37) { CALL(mxfp8_wide); }
  CALL(mxfp8_sm120);
#undef CALL
}

at::Tensor gemm(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,
               std::optional<at::Tensor> output,std::optional<at::Tensor> workspace,
               int64_t tactic,int64_t splits,int64_t swizzle,int64_t sms) {
  auto c=resolve(a,b,tactic,splits,swizzle,sms);
  TORCH_CHECK(a.is_cuda(), "Expected CUDA matrices");
  c10::cuda::CUDAGuard guard(a.device());
  auto out=output.has_value()?*output:at::empty({a.size(0),b.size(0)},a.options().dtype(at::kBFloat16));
  dispatch_out(a,b,sa,sb,out,c,workspace,false);
  return out;
}
std::optional<at::Tensor> allocate_workspace(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,
               at::Tensor const& out,int64_t tactic,int64_t splits,int64_t swizzle,int64_t sms) {
  auto c=resolve(a,b,tactic,splits,swizzle,sms);
  TORCH_CHECK(a.is_cuda(), "Expected CUDA matrices");
  c10::cuda::CUDAGuard guard(a.device());
  auto size=dispatch_out(a,b,sa,sb,out,c,std::nullopt,true);
  if (!size) return std::nullopt;
  return at::zeros({size},a.options().dtype(at::kByte));
}
std::tuple<at::Tensor,std::optional<at::Tensor>,std::vector<int64_t>> prepare(
               at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,
               std::optional<at::Tensor> output,int64_t tactic,int64_t splits,int64_t swizzle,int64_t sms) {
  auto c=resolve(a,b,tactic,splits,swizzle,sms);
  TORCH_CHECK(a.is_cuda(), "Expected CUDA matrices");
  c10::cuda::CUDAGuard guard(a.device());
  auto out=output.has_value()?*output:at::empty({a.size(0),b.size(0)},a.options().dtype(at::kBFloat16));
  auto workspace=allocate_workspace(a,b,sa,sb,out,c.tactic,c.splits,c.swizzle,c.sms);
  return {out,workspace,config_values(c)};
}
at::Tensor gemm_from_float(at::Tensor const& input,at::Tensor const& b,at::Tensor const& sb) {
  // Reuse the same native quantizer as W6A8. No MXFP6 GEMM is involved.
  static auto quantize=c10::Dispatcher::singleton().findSchemaOrThrow("mxfp6::quantize_mxfp8", "")
      .typed<std::tuple<at::Tensor,at::Tensor>(at::Tensor const&)>();
  auto quantized=quantize.call(input);
  auto a=std::get<0>(quantized).view({input.size(0),input.size(1)}).view(at::ScalarType::Float8_e4m3fn);
  return gemm(a,b,std::get<1>(quantized),sb,std::nullopt,std::nullopt,-1,1,1,0);
}
void begin_workspace_planning(at::Tensor const& anchor) { workspace_pool().begin_workspace_planning_cuda(anchor); }
c10::Dict<std::string,int64_t> finalize_workspace_planning(at::Tensor const& anchor) { return workspace_pool().finalize_workspace_planning_cuda(anchor); }
c10::Dict<std::string,int64_t> workspace_stats(at::Tensor const& anchor) { return workspace_pool().workspace_stats_cuda(anchor); }
bool workspace_barriers_zero(at::Tensor const& anchor) { return workspace_pool().workspace_barriers_zero_cuda(anchor); }
} // namespace mxfp8_runtime
#undef MXFP8_ARGS

TORCH_LIBRARY(mxfp8_sm120,m) {
  m.def("gemm(Tensor a, Tensor b, Tensor sa, Tensor sb, Tensor(a!)? out=None, Tensor? workspace=None, int tactic=-1, int splits=1, int swizzle=1, int sms=0) -> Tensor(a!)");
  m.def("prepare(Tensor a, Tensor b, Tensor sa, Tensor sb, Tensor(a!)? out=None, int tactic=-1, int splits=1, int swizzle=1, int sms=0) -> (Tensor(a!), Tensor?, int[])");
  m.def("allocate_workspace(Tensor a, Tensor b, Tensor sa, Tensor sb, Tensor out, int tactic, int splits=1, int swizzle=1, int sms=0) -> Tensor?");
  m.def("gemm_from_float(Tensor input, Tensor b, Tensor sb) -> Tensor");
  m.def("begin_workspace_planning(Tensor anchor) -> ()");
  m.def("finalize_workspace_planning(Tensor anchor) -> Dict(str, int)");
  m.def("workspace_stats(Tensor anchor) -> Dict(str, int)");
  m.def("workspace_barriers_zero(Tensor anchor) -> bool");
  m.def("select_config(int m, int n, int k) -> int[]", &mxfp8_runtime::selected_config);
  m.def("tactics(bool stream_k_only=False) -> int[]", &mxfp8_runtime::tactics);
  m.def("policy_sha256() -> str", &mxfp8_runtime::policy_hash);
}
TORCH_LIBRARY_IMPL(mxfp8_sm120,CUDA,m) {
  m.impl("gemm",mxfp8_runtime::gemm);
  m.impl("prepare",mxfp8_runtime::prepare);
  m.impl("allocate_workspace",mxfp8_runtime::allocate_workspace);
  m.impl("gemm_from_float",mxfp8_runtime::gemm_from_float);
  m.impl("begin_workspace_planning",mxfp8_runtime::begin_workspace_planning);
  m.impl("finalize_workspace_planning",mxfp8_runtime::finalize_workspace_planning);
  m.impl("workspace_stats",mxfp8_runtime::workspace_stats);
  m.impl("workspace_barriers_zero",mxfp8_runtime::workspace_barriers_zero);
}
