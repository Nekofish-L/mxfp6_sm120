#pragma once
#include <cstdint>

namespace mxfp8_runtime {
enum class Backend {
  invalid, core, direct, dynamic, extra, wide, release_core, release_direct, gemv,
#define MXFP8_FAMILY(NAME, BASE, COUNT, RETURN) NAME,
#include "mxfp8_gemm/registry.def"
#undef MXFP8_FAMILY
};
struct Tactic {
  Backend backend = Backend::invalid;
  int64_t argument = 0;
  bool stream_k = false;
};
// Single authority for availability, scheduler class and routing.
inline Tactic decode_tactic(int64_t t) {
#define MXFP8_FAMILY(NAME, BASE, COUNT, RETURN) \
  if (t >= BASE && t < BASE + COUNT) return {Backend::NAME, t-BASE};
#include "mxfp8_gemm/registry.def"
#undef MXFP8_FAMILY
  if (t < 0 || t >= 1600) return {};
  const auto base = t % 100;
  const bool stream_k = (base >= 12 && base <= 16) || base == 39 || base == 40 || base == 42 || base == 51 || base == 57;
  if (t >= 1300) {
    if ((base >= 43 && base <= 50) || (base >= 58 && base <= 69) || (base >= 86 && base <= 95))
      return {Backend::release_direct, t};
    return {};
  }
  if (t >= 800 && t < 1100)
    return base < 33 && base != 30 ? Tactic{Backend::release_core, t, stream_k} : Tactic{};
  const bool replicated = t >= 300 && t < 500;
  if ((t >= 100 && !replicated) || base >= 96 || base == 30 || (base >= 80 && base < 86)) return {};
  if (base >= 33 && base <= 36)
    return replicated ? Tactic{} : Tactic{Backend::gemv, int64_t(1) << (base-33)};
  const auto backend = base >= 70 && base <= 79 ? Backend::dynamic :
                       base >= 51 && base <= 57 ? Backend::extra :
                       base >= 43 ? Backend::direct : base >= 37 ? Backend::wide : Backend::core;
  return {backend, t, stream_k};
}
} // namespace mxfp8_runtime
