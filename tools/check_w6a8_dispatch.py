"""CPU regression fingerprint of effective W6A8 launch choices.

Compile the real host policy/launcher with tensor launches replaced by signatures.
The golden fingerprint was recorded before separating selection from execution.
It covers all M through 2050, large-M boundaries, both dtypes, Stream-K modes,
PDL modes and several SM counts; no CUDA compilation or GPU is required.
"""

from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(source):
    helpers = source[
        source.index("int64_t round_up(") : source.index("void check_input(")
    ]
    launches = source[
        source.index("enum class W6A8Kernel128") : source.index(
            "bool set_w6a8_config_cuda("
        )
    ]
    start = source.find("W6A8LaunchConfig select_w6a8_config(")
    if start < 0:
        start = source.index("at::Tensor launch_w6a8_policy(")
    policy = source[start : source.index("at::Tensor gemm_cuda_impl(")]
    kernels = sorted(
        set(
            re.findall(
                r"(?:normal|swapped|kernels)::(KernelW6A8_\w+)", launches + policy
            )
        )
    )
    prelude = r"""
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) if (!(ok)) throw std::runtime_error("invalid config")
namespace at { using Tensor = std::string; constexpr int kHalf=0; }
int current_output_dtype=0;
bool stream_k=true;
bool stream_k_enabled() { return stream_k; }
enum class RasterOrder { Heuristic, AlongM, AlongN };
struct W6A8LaunchConfig { int64_t config_id, swizzle, raster_order; };
bool find_w6a8_override(int, int64_t, int64_t, int64_t, int, W6A8LaunchConfig&) { return false; }
template<class K> at::Tensor launch_normal(at::Tensor const&, at::Tensor const&,
 at::Tensor const&, at::Tensor const&, int64_t, int64_t, int64_t, double, int,
 int swizzle, RasterOrder raster, int splits, int sms, bool pdl) {
 return std::string(K::name)+":"+std::to_string(swizzle)+":"+
 std::to_string(int(raster))+":"+std::to_string(splits)+":"+
 std::to_string(sms)+":"+std::to_string(pdl);
}
template<class K, class... Args> at::Tensor launch_swapped(Args... args) {
 return launch_normal<K>(args...);
}
"""
    for ns in ("normal", "swapped"):
        prelude += "namespace mxfp6_gemm::" + ns + " {\n"
        for kernel in kernels:
            prelude += f'struct {kernel} {{ static constexpr char const* name="{ns}::{kernel}"; }};\n'
        prelude += "}\n"
    main = r"""
int main() {
 uint64_t hash=14695981039346656037ULL, count=0;
 std::vector<int> ms;
 for(int m=1;m<=2050;++m) ms.push_back(m);
 for(int m: {4095,4096,4097,8191,8192,8193}) ms.push_back(m);
 for(bool sk: {false,true}) for(int dtype: {0,1}) for(bool pdl: {false,true})
 for(int sms: {1,64,132,148,160}) for(int m: ms)
 for(auto nk: std::vector<std::pair<int,int>>{{8192,5120},{5120,3072},
 {7168,5120},{17408,5120},{5120,8704},{128,128},{136,2048},
 {4096,4096},{16384,8192},{5120,5120}}) {
  stream_k=sk; current_output_dtype=dtype;
  auto sig=launch_w6a8_policy("","","","",m,nk.first,nk.second,1.,0,sms,pdl);
  for(unsigned char c:sig) { hash^=c; hash*=1099511628211ULL; }
  hash^=255; hash*=1099511628211ULL; ++count;
 }
 std::cout << count << " " << hash << "\n";
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        cpp = Path(tmp) / "policy.cpp"
        exe = Path(tmp) / "policy"
        cpp.write_text(prelude + helpers + launches + policy + main)
        subprocess.run(
            ["c++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True
        )
        return subprocess.check_output([str(exe)], text=True).strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, nargs="?", default=ROOT / "csrc/torch_extension.cu"
    )
    args = parser.parse_args()
    actual = fingerprint(args.source.read_text())
    assert actual == "822400 12076421243349506047", actual
    print("PASS", actual)
