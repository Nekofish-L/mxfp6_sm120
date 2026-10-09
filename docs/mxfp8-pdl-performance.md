# MXFP8 PDL 扩展与实际性能验证

验证日期：2026-10-08。

## 结论

PDL 在这台 RTX 5090 上能缩短部分依赖链的 GPU 时间，但收益取决于 batch、kernel 和工作负载。完整服务在4/16并发下吞吐约提升2.1%/1.6%；M≤128在64/128并发下相对关闭仅+0.34%/−0.06%，未证明较高并发存在稳定增益。不能把单个算子的加速百分比直接当作服务吞吐收益。

已移除原生显式 PDL API 的 M≤32 限制，补齐较大 batch 使用的 vector-store/TMA eightwarp 路径。性能扫描到 M=2048，正确性覆盖到 M=2049；服务接入新增可配置的行数上限，测试 M≤128。接入默认仍为 PDL 关闭、启用后的上限 32，扩展需要显式设置。

## 三组端到端 HTTP 服务对照

同一 Qwen3.5-4B MXFP8 checkpoint、同一原生库、同一测试 GPU、同一服务配置，只改变 Mach PDL 开关及上限。分别新启动两次服务，顺序为 off → 32 → 128 → 128 → 32 → off；后半程并发点反向执行。下表吞吐取两个服务生命周期的均值，括号为两个原始样本。

| 并发 | PDL 关闭 tok/s | M≤32 tok/s | M≤128 tok/s | 32 相对关闭 | 128 相对关闭 | 128 相对32 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 852.31 (852.21, 852.41) | 870.01 (870.75, 869.27) | 869.82 (870.08, 869.55) | +2.08% | +2.05% | -0.02% |
| 16 | 2769.84 (2769.67, 2770.01) | 2813.56 (2816.81, 2810.31) | 2814.94 (2813.67, 2816.20) | +1.58% | +1.63% | +0.05% |
| 32 | 4203.76 (4195.13, 4212.39) | 4227.66 (4217.61, 4237.70) | 4239.30 (4230.01, 4248.58) | +0.57% | +0.85% | +0.28% |
| 64 | 5178.57 (5171.38, 5185.75) | 5149.25 (5155.80, 5142.70) | 5195.95 (5188.78, 5203.13) | -0.57% | +0.34% | +0.91% |
| 128 | 6075.12 (6042.47, 6107.77) | 6048.58 (6038.18, 6058.98) | 6071.19 (6050.45, 6091.93) | -0.44% | -0.06% | +0.37% |

只有两次生命周期采样。接近样本波动幅度的差异应视为未确认收益；不据此自动扩大默认部署范围。并发数指请求数，原生 M 指 token 行数，可能包含 CUDA Graph padding，prefill 时也可能远大于请求并发。

| 并发 | mean TPOT 关闭 / 32 / 128（ms） | mean TTFT 关闭 / 32 / 128（ms） |
|---:|---:|---:|
| 4 | 4.422 / 4.322 / 4.324 | 140.14 / 142.24 / 142.10 |
| 16 | 5.264 / 5.174 / 5.172 | 258.47 / 258.58 / 257.98 |
| 32 | 6.858 / 6.801 / 6.787 | 373.51 / 380.87 / 377.54 |
| 64 | 11.113 / 11.163 / 11.059 | 602.94 / 613.58 / 609.68 |
| 128 | 18.702 / 18.777 / 18.700 | 1101.48 / 1110.56 / 1110.46 |

服务工作负载：每点固定 1024 输入 token、512 输出 token，2×并发个计分请求，另有并发个 128 输出 token 的 warmup 请求；greedy、ignore_eos、固定每请求 seed。共 2928 个计分请求全部成功，1,499,136 个输出 token。所有组的请求契约及输入 token hash 一致。

服务配置：TP1、BF16 激活、MXFP8 权重、FP8 KV cache、TRITON_ATTN、max_num_seqs=128、max_model_len=4096、max_num_batched_tokens=2048、GPU memory utilization=0.80、关闭 prefix cache、启用 NVFP4 LM head（候选数128、max rows32）。Mach backend=native、NORM_QUANT=0、GDN_TP1=0；请求 FUSED_MLP=1、FUSED_GEMMA_NORM=1，日志未证明实际应用的 fusion 数量。本轮使用 single activation GEMM，未启用 dual GEMM。

这是明确指定 token 长度的自定义 streaming HTTP 工作负载，客户端复用 vllm-mach 的 run_phase；不是该项目公开的六点测试协议。模型输入由固定种子生成 token ID，测量服务路径吞吐，不构成真实用户语料或模型任务质量评估。

## 8 层依赖链：从 M=32 扩到 M=128 仍有 GPU 收益

真实 checkpoint 的 8 组不同 MLP 权重，总占用 583,966,720 bytes（约557 MiB），避免反复测单个权重矩阵都驻留 L2 的情况。激活为合成 BF16 输入，链路为 GemmaNorm → gate/up → SwiGLU+量化 → down，逐层传递 residual。使用先规划再冻结的 workspace；每点 9 轮随机顺序的 paired CUDA Graph 测量，每轮20次 replay，下表为 GPU 时间中位数，收益定义为 off/on−1。

| M | 关闭 PDL µs | 开启 PDL µs | GPU 加速 |
|---:|---:|---:|---:|
| 1 | 426.50 | 411.70 | +3.59% |
| 16 | 440.12 | 426.29 | +3.24% |
| 32 | 437.25 | 424.04 | +3.12% |
| 33 | 450.66 | 436.43 | +3.26% |
| 48 | 453.32 | 443.60 | +2.19% |
| 64 | 456.70 | 442.16 | +3.29% |
| 65 | 491.11 | 475.44 | +3.30% |
| 96 | 504.37 | 486.09 | +3.76% |
| 128 | 511.95 | 496.95 | +3.02% |
| 192 | 624.84 | 619.93 | +0.79% |
| 256 | 691.35 | 680.50 | +1.60% |
| 512 | 1297.20 | 1295.56 | +0.13% |
| 1024 | 2255.31 | 2233.80 | +0.96% |
| 2048 | 4417.13 | 4421.84 | -0.11% |

M=33/48/64/65/96/128 仍有约2.2%–3.8%的链路加速；M≥192 收益明显缩小，M=512/1024/2048 的小差异不足以支持普遍加速结论。这条链路覆盖 norm/MLP 依赖，但没有 attention、GDN、LM head 和 HTTP 调度，其百分比应与上面的完整服务表分开解读。

## 单投影重复回放：PDL 存在回退

相同真实权重，重复同一个投影，权重更热。全部56点如下，值为 off/on−1；负数表示 PDL 更慢。原始时延、9轮样本、具体 tactic/splits/swizzle/sms 保存在 JSON。

| M | QKVZ (12288×2560) | gate/up (18432×2560) | down (2560×9216) | out (2560×4096) |
|---:|---:|---:|---:|---:|
| 1 | +9.35% | +3.43% | +1.85% | +10.23% |
| 16 | +4.69% | +2.64% | -12.49% | +13.19% |
| 32 | +0.96% | -7.76% | -19.51% | +14.49% |
| 33 | -0.69% | -1.66% | -20.79% | +10.28% |
| 48 | -4.29% | -1.97% | -18.95% | -7.62% |
| 64 | -2.79% | -1.07% | -20.63% | -16.11% |
| 65 | -2.75% | +0.40% | +5.00% | -3.17% |
| 96 | -4.40% | -0.10% | +4.21% | -2.47% |
| 128 | -5.99% | +1.00% | +10.52% | -7.13% |
| 192 | -2.57% | -1.21% | -5.41% | +6.67% |
| 256 | +0.00% | +0.38% | +7.11% | +6.16% |
| 512 | +0.32% | +0.60% | +3.27% | +0.52% |
| 1024 | +1.67% | +1.33% | +1.52% | +3.75% |
| 2048 | -5.65% | +1.28% | +0.48% | +1.93% |

例如 M=64 的 down 投影下降约20.6%，同时 8 层链路加速约3.3%。PDL 改变 kernel 的启动/资源竞争，热权重重复回放与跨层依赖链的资源使用不同；当前数据未对各回退点进行 profiler 归因。因此“原生支持更多 M”不能等同于“每个 M、每个投影都更快”。

## 实现与正确性修复

原生改动：`gemm_pdl`、`gemm_from_float_pdl`、`quantize_mxfp8_pdl` 不再对 M>32 静默回落普通启动；vector-store 与 TMA eightwarp 使用现有共同 PDL launcher/prologue，读取输入前等待依赖。不改变原有 dispatch tactic 或量化算术。普通 API 仍按原方式启动。

扩展测试暴露了 allocator 复用问题：量化输出 scale allocation 可能复用前一个 GEMM 的临时 Stream-K workspace。旧代码在 dependency wait 之前写 padding；这些写入会覆盖仍在运行的前驱数据。在 M=192/N=256/K=8192 的 Graph 链上复现了4096个异常 BF16 值。修复为先 wait，再写 padding，然后发送 early launch；padding 也是必须遵守依赖的数据访问。

PDL 的作用是提前调度后继 kernel，重叠可独立完成的初始化或启动开销；实际并发取决于资源，不能依赖一定重叠。依赖数据必须完成同步后才能访问。[NVIDIA CUDA PDL 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)。

验证结果：490项原生 pytest、15项 vLLM 接入/缓存 pytest 全部通过；6项目标 case 的 compute-sanitizer memcheck 为0错误；4组 MXFP6 回归验证通过。较大 M、调度边界、FP16/BF16、零输入、变化输入、旁路 stream、CUDA Graph、普通前驱和下游消费均有覆盖。所有56个投影+14个链路点在4次变化输入 replay中 PDL 开/关逐位一致。

完整服务的调度/动态 batch 可能选用不同 tactic。以下记录每个生命周期的响应文本 hash，以 run0 关闭 PDL 为基准匹配请求索引；这与固定形状算子的逐位一致验证属于不同边界。

| 并发 | 每轮请求数 | run0 off | run1 32 | run2 128 | run3 128 | run4 32 | run5 off |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| 16 | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| 32 | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| 64 | 128 | 128 | 128 | 128 | 128 | 128 | 128 |
| 128 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |

本轮所有对应请求的响应文本 hash 均一致，包括关闭 PDL 自身的两轮。这确认了上述固定输入、greedy 工作负载的生成结果一致。

## 接入位置与使用方法

原生修改位于 `${KERNEL_ROOT}`，基于 `4c33247`，当前未提交。

vLLM 接入位于独立 worktree `${RUNTIME_ROOT}`，分支 `<validation-branch>`，基于 `9c73336`，修改未提交；原 vllm-mach 源码未改动，dist 写入本报告。接入移除了失效的 pdl_version 检查，改为检查所需原生 op；行数上限加入编译缓存因子，本轮实际生成3套不同的缓存 key，分别记录 off/32、on/32、on/128。

在该接入 source 环境下可显式启用扩展：

```bash
export VLLM_MACH_MXFP8_PDL=1
export VLLM_MACH_MXFP8_PDL_MAX_ROWS=128
export MXFP6_LIBRARY_PATH="${KERNEL_ROOT}/build/mxfp8/mxfp6_torch.so"
export MXFP8_LIBRARY_PATH="${KERNEL_ROOT}/build/mxfp8/mxfp8_torch.so"
```

设置 source 的 PYTHONPATH 或安装该 worktree，并重启服务使 Graph 捕获和编译缓存按新配置生效。当前证据支持把128作为可选实验上限；高并发吞吐的增量需按目标服务 workload 复测后决定。

## 复现与原始证据

- [paired GPU benchmark](../benchmarks/benchmark_mxfp8_pdl.py)
- [三组 HTTP benchmark](../benchmarks/benchmark_mxfp8_pdl_serving.py)
- [全部样本、环境、source/library SHA256、服务汇总 JSON](../benchmarks/results/mxfp8_pdl_batch_validation.json)
- 本机全部请求、命令、日志：`${RUNTIME_ROOT}/audit/pdl-batch-validation-20261008`。

GPU 基准和服务使用独立测试 GPU，均为 RTX 5090；driver 595.71.05、Torch2.13.0+cu130、CUDA13.0、vLLM0.29.0。未锁 GPU clock；对照采用随机顺序/反向生命周期以减小漂移，仍不能把小差异当作统计显著性。本次验证只覆盖一台机器、一个模型和上述 workload。

```bash
cd "${KERNEL_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=python python3 benchmarks/benchmark_mxfp8_pdl.py \
  --checkpoint "${MODEL_DIR}/model.safetensors" \
  --vllm-source "${RUNTIME_ROOT}/src" --output /tmp/pdl-recheck.json

"${RUNTIME_PYTHON}" benchmarks/benchmark_mxfp8_pdl_serving.py \
  --python "${RUNTIME_PYTHON}" \
  --model "${MODEL_DIR}" \
  --vllm-source "${RUNTIME_ROOT}/src" \
  --client-source "${RUNTIME_ROOT}/src/vllm_mach/mxfp8/benchmark.py" \
  --gpu "${GPU_ID}" --output /tmp/pdl-serving-recheck
```

复现命令中的变量按本地环境设置：`KERNEL_ROOT` 为内核仓库，`RUNTIME_ROOT` 为对应 runtime 源码，`RUNTIME_PYTHON` 为其 Python 环境，`MODEL_DIR` 为模型目录，`AUDIT_ROOT` 为本地审计目录，`PROMPTS_FILE` 为提示 JSON，`GPU_ID` 为测试设备编号。公开报告不包含原始提示、生成内容或私有权重；路径和设备标识已脱敏，内容哈希对应原始测量文件。
