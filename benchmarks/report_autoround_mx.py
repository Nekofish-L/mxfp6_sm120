#!/usr/bin/env python3
"""Render the saved rounding simulation results as Markdown."""
import json
from pathlib import Path
import statistics

p=Path(__file__).resolve().parents[1]/'benchmarks/results/qwen35_2b_autoround_sim'
s=json.loads((p/'summary.json').read_text())
rows=[json.loads(t) for t in (p/'raw.jsonl').read_text().splitlines()]
max_check=max(e['kernel_vs_decoded'] for r in rows for v in r['results'].values() for e in [v['calibration'],v['validation']]+v['tests'])
lines=['# MXFP6 / MXFP8：AutoRound 思路的独立 GEMM 模拟','',
'## 结论','',
'固定 E8M0 scale，仅学习权重向相邻浮点数舍入的方向，可以在具有通道相关性的 fake input 上降低独立测试集误差，且不修改原生 GEMM kernel。独立正态输入上没有实质性泛化收益，单看校准集误差会误判为提升。','',
'## 模拟范围','',
'- Qwen3.5-2B 的 7 类注意力投影各 1 个完整矩阵：第 0 层 in_proj_qkv、in_proj_z、out_proj，以及第 3 层 q/k/v/o_proj。全部 K=2048，N 为 512/2048/4096/6144。没有裁剪权重。',
'- 以现有 MXFP6/MXFP8 checkpoint 初始化，保留 group=32 和 E8M0 scale；原始 BF16 权重提供参考目标。原模型文件未修改。',
'- 激活统一使用动态 MXFP8；校准用其反量化值计算，最终测试调用原生 W6A8/W8A8 GEMM，FP32 累加、BF16 输出。',
'- 模拟 AutoRound 的舍入优化思路：SignSGD + 直通估计器（STE）；不是完整 AutoRound 包，没有 decoder block、scale 或 clipping 优化，也不声称完整 AutoRound 已支持本 MXFP6 checkpoint。',
'- 200 步，初始学习率 0.005，线性衰减；batch=128。校准 1024 行、独立验证 512 行、3 个独立测试种子各 512 行。28 次优化实验，主表每项平均 7 个矩阵 × 3 组测试输入。',
'- 每 10 步用完整校准集和独立验证集分别选择最优舍入，原始 checkpoint 也是候选。测试集不参与参数或步数选择。','',
'## 优化逻辑','',
'对每个权重，在固定 scale 下找到两个相邻合法浮点值，学习偏移 v 来决定舍入：','',
'```text','p = 权重在相邻两个量化值之间的位置','Wq(v) = lower + (upper-lower) * 1[p+v >= 0.5]',
'loss = MSE(Xq @ Wq(v).T, X @ W.T)','用 STE 对舍入求梯度','v = clamp(v - lr * sign(gradient), -0.5, 0.5)','```','',
'初始化保留 checkpoint 的实际舍入，包括中点情况。前向权重始终是合法离散值，优化后重新打包供原 kernel 使用。',
'方法参考：[Optimize Weight Rounding via Signed Gradient Descent](https://aclanthology.org/2024.findings-emnlp.662/)。本实验只模拟其中舍入学习部分。','',
'## 输入分布','',
'1. **独立正态**：每个通道独立 N(0,1)，与此前精度测试同类。',
'2. **相关随机**：64 维正态隐变量映射到 2048 维，加入标准差 0.1 的独立噪声，规范化各通道方差。三个数据集共享分布参数，但样本和种子独立。这是人为构造的相关性，并非真实模型激活。','',
'## 独立测试集：验证集选取结果','','| 输入 | 格式 | 原始误差 | 优化后误差 | 相对下降 |','|---|---|---:|---:|---:|']
for scenario,label in [('gaussian','独立正态'),('correlated','相关随机')]:
 for bits in [6,8]:
  v=s[scenario][str(bits)];b=v['baseline']['mean_test'];t=v['validation_selected']['mean_test']
  lines.append(f'| {label} | MXFP{bits} W{bits}A8 | {b*100:.4f}% | {t*100:.4f}% | {(1-t/b)*100:.3f}% |')
lines+=['',
'独立正态输入下 MXFP6 所有矩阵都选回原权重；MXFP8 的微小变化不构成实质改善。相关输入下所有 7 个矩阵的独立测试误差都降低。','',
'## 仅按校准集选取时的过拟合','','| 格式 | 校准误差：原始 → 优化后 | 测试误差：原始 → 优化后 |','|---|---:|---:|']
for bits in [6,8]:
 v=s['gaussian'][str(bits)];b=v['baseline'];t=v['train_selected']
 lines.append(f"| MXFP{bits} | {b['mean_calibration']*100:.4f}% → {t['mean_calibration']*100:.4f}% | {b['mean_test']*100:.4f}% → {t['mean_test']*100:.4f}% |")
lines+=['',
'忽略激活量化，输入协方差为 Σ 时，期望输出平方误差受 `tr(ΔW Σ ΔWᵀ)` 控制。独立单位方差正态输入的 Σ=I，目标就是权重 Frobenius 误差；固定 scale 时最近舍入已适合此目标。有限校准样本上学习可能追逐样本噪声。',
'相关输入的 Σ 不等于 I，通过调整舍入让不同通道误差在输出上抵消，可以获得收益；权重自身的误差不一定随之减小。','',
'## 训练过程','','下面是反量化 FP32 计算的平均误差；主表使用原生 kernel 的 BF16 输出。','','| 输入 | 格式 | 步数 | 校准误差 | 验证误差 |','|---|---|---:|---:|---:|']
for scenario in ['gaussian','correlated']:
 for bits in [6,8]:
  rr=[r for r in rows if r['scenario']==scenario and r['bits']==bits]
  for step in [0,50,100,150,200]:
   ts=[next(t for t in r['trace'] if t['step']==step) for r in rr]
   lines.append(f"| {scenario} | MXFP{bits} | {step} | {statistics.mean(t['train'] for t in ts)*100:.4f}% | {statistics.mean(t['validation'] for t in ts)*100:.4f}% |")
lines+=['','## 原生 kernel 校验','',
f'所有基线及优化权重均在校准、验证和 3 组测试输入上校验。kernel 相对反量化参考的最大误差为 **{max_check*100:.6f}%**，全部通过相对 RMSE < 0.001 的断言。','',
f"相关输入下，平均改变 {s['correlated']['6']['validation_selected']['mean_changed_weight_fraction']*100:.2f}% 的 MXFP6 权重舍入选择、{s['correlated']['8']['validation_selected']['mean_changed_weight_fraction']*100:.2f}% 的 MXFP8 权重舍入选择。scale、激活量化和 GEMM kernel 保持不变。",'',
'## 逐投影测试结果','','| 输入 | 权重 | 格式 | 基线 | 验证集选出的结果 |','|---|---|---|---:|---:|']
for r in rows:
 b=r['results']['baseline']['mean_test_relative_rmse'];t=r['results']['validation_selected']['mean_test_relative_rmse']
 lines.append(f"| {r['scenario']} | {r['weight'].replace('model.language_model.','')} | MXFP{r['bits']} | {b*100:.4f}% | {t*100:.4f}% |")
lines+=['','## 复现','',
'```bash','CUDA_VISIBLE_DEVICES=0 python3 benchmarks/simulate_autoround_mx.py','python3 benchmarks/report_autoround_mx.py','```','',
'`metadata.json` 记录参数、分布和种子；`raw.jsonl` 记录逐步曲线、权重误差和 kernel 校验；`summary.json` 为汇总。','',
'**实际收益仍需真实激活校准验证。** 本实验说明不改 GEMM、只优化舍入可以有效；相关 fake input 上的收益不能直接视为 Qwen3.5 实际推理收益。','']
(p/'report.md').write_text('\n'.join(lines))
print(p/'report.md')
