#pragma once
#include "mxfp6_gemm/types.hpp"

namespace mxfp6_dispatch {
inline int64_t output_tile_count(int64_t m, int64_t n, int64_t tile_m) {
  return round_up(m, tile_m) / tile_m * (round_up(n, 128) / 128);
}

inline int64_t final_wave_tiles(int64_t tiles, int sm_count) {
  int64_t const tail = tiles % sm_count;
  // A modulo-zero tail is a completely occupied final wave, not an empty one.
  return tail == 0 ? sm_count : tail;
}

// Cache-only serving deliberately keeps an unknown eager-prefill shape on the
// static path.  These two rules cover the repeatable post-graph tail-wave
// holes seen on Qwen3.5-27B TP2 without starting a runtime profile.  Keep the
// scope below the checked-in M=2048 entry: larger prefill limits should use an
// offline cache until they have their own profiler evidence.
inline bool use_eager_prefill_64x128(int64_t m, int64_t n, int64_t k,
                               int sm_count) {
  if (m < 1024 || m >= 2048 || n != 5120 || k != 3072) {
    return false;
  }
  int64_t const tiles_64 = output_tile_count(m, n, 64);
  int64_t const tiles_128 = output_tile_count(m, n, 128);
  return final_wave_tiles(tiles_64, sm_count) >
      final_wave_tiles(tiles_128, sm_count);
}

inline bool use_eager_prefill_stream_k(int64_t m, int64_t n, int64_t k,
                                 int sm_count, bool stream_k) {
  if (!stream_k || m < 1024 || m >= 2048 || k != 5120 ||
      (n != 7168 && n != 8192)) {
    return false;
  }
  int64_t const tail = output_tile_count(m, n, 128) % sm_count;
  // Beyond the generic four-wave cutoff, retain Stream-K only for a very
  // short final wave.  One third of an RTX 5090's SMs is conservative across
  // the measured QKV and GDN-in windows, while avoiding its nearby regressions.
  return tail > 0 && tail <= sm_count / 3;
}

inline Config normalize(Config config, bool stream_k) {
  if (!stream_k) {
    if (config.config_id == 3) config.config_id = 1;
    if (config.config_id == 16) config.config_id = 12;
  }
  return config;
}

inline Config select_builtin(int64_t m, int64_t n, int64_t k, int sm_count,
                             bool half_output, bool stream_k) {
  const bool target_nk = is_target_nk(n, k);
  const auto stream_k_enabled = [stream_k] { return stream_k; };
  const auto select = [&]() -> Config {
#include "mxfp6_gemm/w6a8_exact.inc"
    if (use_eager_prefill_64x128(m, n, k, sm_count)) return {10, 1, 0};
    if (use_eager_prefill_stream_k(m, n, k, sm_count, stream_k)) return {16, 1, 0};
    if (m <= 8) return {1, 4, 0};
    if (m <= 16) return {0, 2, 0};
    if (m <= 64) return {7, 2, 0};
    if (m <= 128) return {11, 2, 0};
    const int64_t tiles = round_up(m, 128) / 128 * (round_up(n, 128) / 128);
    const int64_t tail = tiles % sm_count;
    const bool underfilled = k >= 2048 &&
        ((tiles < sm_count) || (tiles <= 4LL * sm_count && tail > 0 && tail < sm_count / 2));
    return {underfilled ? 16 : 12, underfilled ? 1 : 2, 0};
  };
  return normalize(select(), stream_k);
}
} // namespace mxfp6_dispatch
