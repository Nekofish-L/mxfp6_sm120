#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <optional>
#include <vector>
#include "mxfp8_gemm/validation.hpp"
#include "mxfp8_gemm/workspace.hpp"
#include "mxfp8_gemm/pdl.cuh"

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

namespace mxfp8_bf16_ba { at::Tensor bf16_ba_m16(at::Tensor const& x, at::Tensor const& weight); }

namespace mxfp8_runtime {
struct LaunchConfig { int64_t tactic, splits, swizzle, sms; };
thread_local bool use_pdl = false;
bool pdl_enabled() { return use_pdl; }
struct PdlScope {
  bool previous;
  explicit PdlScope(bool enabled) : previous(use_pdl) { use_pdl = enabled; }
  ~PdlScope() { use_pdl = previous; }
};
WorkspacePool& workspace_pool() { static WorkspacePool pool; return pool; }
}
#include "mxfp8_gemm/dispatch_policy.hpp"

namespace mxfp8_runtime {
enum class TacticFamily {
  Invalid, Core, Direct, Dynamic, Extra, Wide, ReleaseCore, ReleaseDirect, Gemv,
#define MXFP8_FAMILY(NAME, BASE, COUNT, RETURN) NAME,
#include "mxfp8_gemm/registry.def"
#undef MXFP8_FAMILY
};
bool core_tactic(int64_t t) { return t >= 0 && t < 96 && t != 30 && !(t >= 80 && t < 86); }
bool release_direct_base(int64_t t) {
  return (t >= 43 && t <= 50) || (t >= 58 && t <= 69) || (t >= 86 && t <= 95);
}
TacticFamily tactic_family(int64_t t) {
#define MXFP8_FAMILY(NAME, BASE, COUNT, RETURN) if (t >= BASE && t < BASE + COUNT) return TacticFamily::NAME;
#include "mxfp8_gemm/registry.def"
#undef MXFP8_FAMILY
  if (t >= 1300 && t < 1600)
    return release_direct_base(t % 100) ? TacticFamily::ReleaseDirect : TacticFamily::Invalid;
  if (t >= 800 && t < 1100)
    return t % 100 < 33 && t % 100 != 30 ? TacticFamily::ReleaseCore : TacticFamily::Invalid;
  auto base = t >= 300 && t < 500 ? t % 100 : t;
  if (!core_tactic(base) || (t != base && base >= 33 && base <= 36)) return TacticFamily::Invalid;
  if (base >= 33 && base <= 36) return TacticFamily::Gemv;
  if (base >= 70 && base <= 79) return TacticFamily::Dynamic;
  if (base >= 51 && base <= 57) return TacticFamily::Extra;
  if (base >= 43) return TacticFamily::Direct;
  if (base >= 37) return TacticFamily::Wide;
  return TacticFamily::Core;
}
bool valid_tactic(int64_t t) { return tactic_family(t) != TacticFamily::Invalid; }
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
  // The public entry owns the device guard; validate once before family dispatch.
  mxfp8_common::validate(a,b,sa,sb,out,query);
  mxfp8_common::validate_scheduler(c.splits,c.swizzle);
  auto t=c.tactic;
#define CALL(NS) return NS::mm_out(a,b,sa,sb,out,t,c.splits,c.swizzle,workspace,query,c.sms)
  switch (tactic_family(t)) {
    case TacticFamily::ReleaseDirect: { CALL(mxfp8_release_direct); }
    case TacticFamily::ReleaseCore: { CALL(mxfp8_release_core); }
#define MXFP8_FAMILY(NAME, BASE, COUNT, RETURN) \
    case TacticFamily::NAME: if (!query) mxfp8_##NAME::mm_out(a,b,sa,sb,out,t-BASE); return 0;
#include "mxfp8_gemm/registry.def"
#undef MXFP8_FAMILY
    case TacticFamily::Gemv:
      if (!query) mxfp8_sm120::gemv_out(a,b,sa,sb,out,int64_t(1) << (t-33));
      return 0;
    case TacticFamily::Dynamic: { CALL(mxfp8_dynamic); }
    case TacticFamily::Extra: { CALL(mxfp8_extra); }
    case TacticFamily::Direct: { CALL(mxfp8_direct); }
    case TacticFamily::Wide: { CALL(mxfp8_wide); }
    case TacticFamily::Core: { CALL(mxfp8_sm120); }
    case TacticFamily::Invalid: TORCH_CHECK(false, "Unknown MXFP8 tactic: ", t);
  }
  TORCH_CHECK(false, "Unreachable MXFP8 tactic family");
#undef CALL
}

at::Tensor gemm(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,
               std::optional<at::Tensor> output,std::optional<at::Tensor> workspace,
               int64_t tactic,int64_t splits,int64_t swizzle,int64_t sms,bool enable_pdl) {
  PdlScope scope(enable_pdl);
  auto c=resolve(a,b,tactic,splits,swizzle,sms);
  TORCH_CHECK(a.is_cuda(), "Expected CUDA matrices");
  c10::cuda::CUDAGuard guard(a.device());
  auto out=output.has_value()?*output:at::empty({a.size(0),b.size(0)},a.options().dtype(at::kBFloat16));
  dispatch_out(a,b,sa,sb,out,c,workspace,false);
  return out;
}
std::optional<at::Tensor> allocate_workspace(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb,
               std::optional<at::Tensor> output,int64_t tactic,int64_t splits,int64_t swizzle,int64_t sms) {
  auto out = output.value_or(at::Tensor{});
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
at::Tensor gemm_pdl(at::Tensor const& a,at::Tensor const& b,at::Tensor const& sa,at::Tensor const& sb) {
  return gemm(a,b,sa,sb,std::nullopt,std::nullopt,-1,1,1,0,true);
}
template<bool Pdl>
at::Tensor gemm_from_float_impl(at::Tensor const& input,at::Tensor const& b,at::Tensor const& sb) {
  // Reuse the same native quantizer as W6A8. No MXFP6 GEMM is involved.
  static auto quantize=c10::Dispatcher::singleton().findSchemaOrThrow(
      Pdl ? "mxfp6::quantize_mxfp8_pdl" : "mxfp6::quantize_mxfp8", "")
      .typed<std::tuple<at::Tensor,at::Tensor>(at::Tensor const&)>();
  auto quantized=quantize.call(input);
  auto a=std::get<0>(quantized).view({input.size(0),input.size(1)}).view(at::ScalarType::Float8_e4m3fn);
  return gemm(a,b,std::get<1>(quantized),sb,std::nullopt,std::nullopt,-1,1,1,0,Pdl);
}
at::Tensor gemm_from_float(at::Tensor const& input,at::Tensor const& b,at::Tensor const& sb,bool enable_pdl) {
  return enable_pdl ? gemm_from_float_impl<true>(input,b,sb)
                    : gemm_from_float_impl<false>(input,b,sb);
}
at::Tensor gemm_from_float_pdl(at::Tensor const& input,at::Tensor const& b,at::Tensor const& sb) {
  return gemm_from_float_impl<true>(input,b,sb);
}
void begin_workspace_planning(at::Tensor const& anchor) { workspace_pool().begin_workspace_planning_cuda(anchor); }
c10::Dict<std::string,int64_t> finalize_workspace_planning(at::Tensor const& anchor, bool per_stream) { return workspace_pool().finalize_workspace_planning_cuda(anchor, per_stream); }
c10::Dict<std::string,int64_t> workspace_stats(at::Tensor const& anchor) { return workspace_pool().workspace_stats_cuda(anchor); }
bool workspace_barriers_zero(at::Tensor const& anchor) { return workspace_pool().workspace_barriers_zero_cuda(anchor); }
} // namespace mxfp8_runtime
#undef MXFP8_ARGS

TORCH_LIBRARY(mxfp8_sm120,m) {
  m.def("bf16_ba_m16(Tensor x, Tensor weight) -> Tensor");
  m.def("gemm(Tensor a, Tensor b, Tensor sa, Tensor sb, Tensor(a!)? out=None, Tensor? workspace=None, int tactic=-1, int splits=1, int swizzle=1, int sms=0, bool use_pdl=False) -> Tensor(a!)");
  m.def("prepare(Tensor a, Tensor b, Tensor sa, Tensor sb, Tensor(a!)? out=None, int tactic=-1, int splits=1, int swizzle=1, int sms=0) -> (Tensor(a!), Tensor?, int[])");
  m.def("allocate_workspace(Tensor a, Tensor b, Tensor sa, Tensor sb, Tensor? out=None, int tactic=-1, int splits=1, int swizzle=1, int sms=0) -> Tensor?");
  m.def("gemm_from_float(Tensor input, Tensor b, Tensor sb, bool use_pdl=False) -> Tensor");
  m.def("gemm_pdl(Tensor a, Tensor b, Tensor sa, Tensor sb) -> Tensor");
  m.def("gemm_from_float_pdl(Tensor input, Tensor b, Tensor sb) -> Tensor");
  m.def("begin_workspace_planning(Tensor anchor) -> ()");
  m.def("finalize_workspace_planning(Tensor anchor, bool per_stream=False) -> Dict(str, int)");
  m.def("workspace_stats(Tensor anchor) -> Dict(str, int)");
  m.def("workspace_barriers_zero(Tensor anchor) -> bool");
  m.def("select_config(int m, int n, int k) -> int[]", &mxfp8_runtime::selected_config);
  m.def("tactics(bool stream_k_only=False) -> int[]", &mxfp8_runtime::tactics);
  m.def("policy_sha256() -> str", &mxfp8_runtime::policy_hash);
}
TORCH_LIBRARY_IMPL(mxfp8_sm120,CUDA,m) {
  m.impl("bf16_ba_m16",mxfp8_bf16_ba::bf16_ba_m16);
  m.impl("gemm",mxfp8_runtime::gemm);
  m.impl("prepare",mxfp8_runtime::prepare);
  m.impl("allocate_workspace",mxfp8_runtime::allocate_workspace);
  m.impl("gemm_from_float",mxfp8_runtime::gemm_from_float);
  m.impl("gemm_pdl",mxfp8_runtime::gemm_pdl);
  m.impl("gemm_from_float_pdl",mxfp8_runtime::gemm_from_float_pdl);
  m.impl("begin_workspace_planning",mxfp8_runtime::begin_workspace_planning);
  m.impl("finalize_workspace_planning",mxfp8_runtime::finalize_workspace_planning);
  m.impl("workspace_stats",mxfp8_runtime::workspace_stats);
  m.impl("workspace_barriers_zero",mxfp8_runtime::workspace_barriers_zero);
}
