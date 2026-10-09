"""Validate complete-layer coverage and summarize natural-cache inference runs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from public_metadata import redact_local_metadata
import statistics as stats


def collect(directory: Path, projections: bool):
    grouped = defaultdict(list)
    for path in directory.glob("steps-*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["warmup"] or not row["pure_decode"] or row["requests"] != row["batch"]:
                continue
            if projections:
                ids = {(p["kind"],p["layer"]) for p in row["projections"]}
                required = {("gateup",i) for i in range(32)} | {("qkvz",i) for i in range(24)}
                assert ids == required and len(row["projections"]) == 56, (directory,row["batch"],ids)
                physical = {p["m"] for p in row["projections"]}
                assert len(physical) == 1
                row["physical_m"] = physical.pop()
            else:
                assert row["backbone_ms"] is not None and row["physical_m"] is not None
            grouped[(row["context"],row["batch"],row["repeat"])].append(row)
    points = defaultdict(dict)
    for (context,batch,repeat), rows in grouped.items():
        assert len(rows) >= 8, (directory,context,batch,len(rows))
        assert len({r["physical_m"] for r in rows}) == 1
        phases = {"first":rows[:1],"steady":rows[3:-2]}
        for phase, selected in phases.items():
            values = {"physical_m":rows[0]["physical_m"], "steps":len(selected)}
            if projections:
                values["instrumented_backbone_ms"] = stats.median(r["backbone_ms"] for r in selected)
                for kind in ("qkvz","gateup"):
                    for metric in ("quant_us","gemm_us","total_us"):
                        values[f"{kind}_{metric}"] = stats.median(
                            stats.fmean(p[metric] for p in r["projections"] if p["kind"] == kind)
                            for r in selected)
            else:
                values["backbone_ms"] = stats.median(r["backbone_ms"] for r in selected)
            points[(context,batch)].setdefault(phase, {})[repeat] = values
    for key, phases in points.items():
        assert set(phases["steady"]) == {0,1}, (directory,key,phases)
    return points


def average(repeats, metric):
    return stats.fmean(r[metric] for r in repeats.values())


def identity(directory):
    run = json.loads((directory/"run.json").read_text())
    receipts = []
    for path in directory.glob("worker-*.json"):
        r = json.loads(path.read_text())
        receipts.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                         "profile":r["profile"],"phase":r["phase"],"versions":r["versions"],
                         "runtime_sources":r["runtime_sources"],"binary":r["native"]["binary"],
                         "gateup_layers":r["dual"]["gateup_layer_count"],
                         "qkvz_layers":r["dual"]["qkvz_layer_count"]})
    assert len(receipts) == 1
    return {"directory":str(directory),"run":run,"receipt":receipts[0]}


def render_markdown(artifact, stem):
    records=artifact["points"]
    short=[r for r in records if r["context"]==256]
    long=[r for r in records if r["context"]==3000]
    s=artifact["summary"]["256"]
    percent=lambda v:f"{(v-1)*100:+.1f}%"
    text=["# Dual GEMM 对比 single：真实模型推理的自然缓存，BS=1–128", "",
          "测量日期：2026-10-09。模型：Qwen3.5-4B，BF16 转换的 MXFP8 checkpoint；RTX 5090。", "",
          f"排除量化时间，真实推理稳态下 QKVZ 的 GEMM 延迟几何平均变化为 **{percent(s['steady_qkvz_gemm_us_ratio']['geomean'])}**，"
          f"gate/up 为 **{percent(s['steady_gateup_gemm_us_ratio']['geomean'])}**。"
          f"作为次要交叉验证，当前未融合路径移除投影内部计时节点后的完整 backbone 前向为 **{percent(s['steady_backbone_ratio']['geomean'])}**。"
          "这里正数表示 dual 更慢；dual 的价值应由精度需求评估，不能把双路累加当作已验证的加速方案。", "",
          f"![真实推理位置的 GEMM-only 对比]({stem}.svg)", "",
          "## 实际冷热计算的定义", "",
          "本次用 vLLM 的完整生成过程建立缓存状态：真实 embedding、32 层模型、GDN/attention、MLP、KV cache、部署端 LM head 和采样均实际执行。"
          "投影直接接收模型中产生的 BF16 激活。没有随机激活，没有反复只回放同一权重，没有清空 L2，也没有在同一个投影处先执行一组再执行另一组。", "",
          "`first` 是所有请求完成 prefill 后的第一个完整 batch decode 步；`steady` 是随后连续 decode，去掉前 3 步和最后 2 步。"
          "二者保留实际的缓存状态，但不代表硬件 L2 的全冷/全热端点。本次没有测量 L2 命中率。"
          "模型各层的权重工作集远大于 96 MiB L2，连续 decode 也不能等同于单权重驻留 L2 的 hot microbenchmark。", "",
          "## 对照与测量边界（主比较只计 GEMM）", "",
          "- 按用户的激活函数+量化融合计划，主比较排除量化，只使用量化完成后的 GEMM 区间；本次没有实现这项融合，完整前向数字不代表未来融合后的部署收益。",
          "- 保持 `qwen35-4b-mxfp8-champion-v1` 的 runtime 补丁、GDN、BA/QKVZ 并行、KV FP8、编译和 Graph 策略一致。依赖版本与 runtime 源码 SHA256 校验均通过。",
          "- single 使用部署端 `mxfp8_e4m3_quantize` 加普通 native GEMM；dual 使用 high/residual 两路量化加当前仓库原生 dual GEMM。两边 PDL 均关闭。",
          "- 测量 hook 将 32 个 gate/up 和 24 个 QKVZ 在 M=1–128 切到对应实验 arm。发布的 champion 默认仍是 gate/up M32、QKVZ M32/M64；本次验证的是扩展路径，不表示线上默认策略已变更。",
          "- 投影时间来自 CUDA Graph 内的 external CUDA events，真实 replay 会更新时间戳。逐步核对全部 56 个投影以及实际 M。",
          "- 投影数字是事件包围的算子区间，包含内部计时节点的影响，并非 CUPTI 提取的纯 kernel duration；无内部计时节点的完整前向用于检查是否存在明显的计时观察效应，融合后的最终前向仍需在融合实现上测量。",
          "- 另起两组生命周期，移除全部投影内部事件，仅在完整 Graph replay 外计时，交叉验证完整 backbone 性能。该时间不包含 LM head、采样、CPU 调度或 HTTP；这些步骤仍照常执行并影响下一步缓存。",
          "- ISL256 全测 128 个请求 batch，OSL32；每点正向/反向各一次。各重复内先取稳态步的中位数，再对两次重复求平均。投影时间先对该类所有层取平均。",
          "- ISL3000 补测实际长对话，OSL384，使 chunked prefill 后仍有足够的完整 batch decode 步。使用 7 个代表 batch，正向/反向各一次。",
          "- ISL256 使用 62 条足够长的实际对话截取前缀；ISL3000 使用数据集唯一一条 3000-token 对话，独立请求重复使用，prefix cache 关闭。两组使用相同提示和采样参数，但生成 token 可以因数值路径不同而变化。", "",
          "## 部署 Graph 与原生 M 的关系", "",
          "请求 BS=1–128 在部署 profile 下映射到 19 个实际 M：1、2、4、8、16、24、…、128。"
          "例如 BS3→M4、BS33→M40。这是部署端 Graph padding；dual 内核自身直接接收实际 M，未额外复制、拆分或补齐。"
          "当前 native API 对每个整数 M=1–128 的支持与此前正确性测试保留；真实部署测量应按下表的实际 M 解读。", "",
          "single 使用仓库现有的按形状 tactic dispatch，dual 在 M32 之后切换 tile family。配置切换会造成曲线跃变，特别是 QKVZ 的请求 BS97–112 与 BS113–128；本轮没有重新调优 single 或 dual 的 dispatch。", "",
          "## 首步与稳态汇总", "",
          "| 阶段 | QKVZ GEMM D/S | gate/up GEMM D/S | 无内部事件 backbone D/S |",
          "|---|---:|---:|---:|"]
    for phase,label in [("first","prefill 后首个完整 batch decode"),("steady","连续 decode 稳态")]:
        text.append(f"| {label} | {s[phase+'_qkvz_gemm_us_ratio']['geomean']:.3f} | {s[phase+'_gateup_gemm_us_ratio']['geomean']:.3f} | {s[phase+'_backbone_ratio']['geomean']:.3f} |")
    text += ["", "## 不同 batch 范围", "",
             "| 请求 BS 范围 | 稳态 QKVZ D/S | 稳态 gate/up D/S | 无内部事件 backbone D/S |",
             "|---|---:|---:|---:|"]
    for name,values in artifact["batch_ranges"].items():
        text.append(f"| {name} | {values['steady_qkvz_gemm_us_ratio']:.3f} | {values['steady_gateup_gemm_us_ratio']:.3f} | {values['steady_backbone_ratio']:.3f} |")
    text += ["", "## 代表点：稳态、自然缓存", "",
          "GEMM 为单个投影的全层平均微秒，排除量化；backbone 为移除投影计时节点后的完整前向毫秒。", "",
          "| 请求 BS | 实际 M | QKVZ single/dual μs | gate/up single/dual μs | backbone single/dual ms | backbone D/S |",
          "|---:|---:|---:|---:|---:|---:|"]
    for r in short:
        if r["batch"] in (1,2,4,8,16,24,32,40,48,64,80,96,112,128):
            text.append(f"| {r['batch']} | {r['physical_m']} | {r['single_steady_qkvz_gemm_us']:.2f}/{r['dual_steady_qkvz_gemm_us']:.2f} | {r['single_steady_gateup_gemm_us']:.2f}/{r['dual_steady_gateup_gemm_us']:.2f} | {r['single_steady_backbone_ms']:.3f}/{r['dual_steady_backbone_ms']:.3f} | {r['steady_backbone_ratio']:.3f} |")
    text += ["", "## 3000-token 上下文：完整前向交叉验证", "",
             "| 请求 BS | 实际 M | first single/dual ms | first D/S | steady single/dual ms | steady D/S |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in long:
        text.append(f"| {r['batch']} | {r['physical_m']} | {r['single_first_backbone_ms']:.3f}/{r['dual_first_backbone_ms']:.3f} | {r['first_backbone_ratio']:.3f} | {r['single_steady_backbone_ms']:.3f}/{r['dual_steady_backbone_ms']:.3f} | {r['steady_backbone_ratio']:.3f} |")
    text += ["", "## BS=1–128 全部结果", "",
             "D/S=dual 延迟÷single 延迟，小于 1 才是 dual 更快。绝对延迟和 GEMM-only 分解见 CSV/JSON。", "",
             "| BS | M | first QKVZ GEMM D/S | first gate/up GEMM D/S | steady QKVZ GEMM D/S | steady gate/up GEMM D/S | steady backbone D/S |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in short:
        text.append(f"| {r['batch']} | {r['physical_m']} | {r['first_qkvz_gemm_us_ratio']:.3f} | {r['first_gateup_gemm_us_ratio']:.3f} | {r['steady_qkvz_gemm_us_ratio']:.3f} | {r['steady_gateup_gemm_us_ratio']:.3f} | {r['steady_backbone_ratio']:.3f} |")
    text += ["", "## 对当前实现的判断", "",
             "当前 dual 已原生支持 M=1–128，但范围扩展没有使所有 M 都成为高效配置。M>32 沿用 128×32×128 的两阶段双累加器 tile family；两路 MMA 和双 accumulator 的资源成本仍然存在。"
             "测量显示较小 batch 可以用较低的 GEMM 代价采用双路表示，较大 batch 的计算代价显著增加。"
             "仅融合激活函数和量化会移除前置开销，不会自动消除本报告已经排除量化后的 GEMM 差距。"
             "若目标是加速，应继续针对较大 M 调整 dual 的 tile/资源配置，再在同样的实际推理位置验证；本次不能据此给出全范围默认启用的性能理由。", "",
             "## 证据和复现", "",
             f"- [全部延迟 CSV]({stem}.csv)、[结果和环境 JSON]({stem}.json)、[曲线]({stem}.svg)。",
             "- native 实现：`mxfp6_sm120/python/mxfp6/mxfp8_dual.py`；完整推理 harness：`benchmarks/benchmark_mxfp8_dual_inference.py`；覆盖检查和汇总：`benchmarks/report_mxfp8_dual_inference.py`。",
             "- 原始逐步/逐层数据、生成 token 和部署审计文件保留在本地审计目录，不随公开报告分发。JSON 中的路径及设备标识已脱敏；内容哈希对应原始测量输入和文件。运行分组：`single-v2`、`dual`、`dual-forward`、`single-forward`。",
             "- 从 vllm0.29.0 和 b12x1.2.6 wheel 解压到独立 site，复制 main06a08a3 的 `vllm_mach`，使用 `mxfp8.install.install_profile` 应用已校验 runtime 补丁。未改全局 vLLM 或正在运行的服务。",
             "- 隔离 profile 的 `install_worker_hook` 调用测量脚本的 `install_measurement()`，再执行原 profile hook，最后 `wrap_worker()`。单独启动进程，设置 `MX8_INFERENCE_ARM=single/dual`；完整前向验证设置 `MX8_INFERENCE_EVENTS=0`。",
             "- 所有运行使用同一测试 GPU；未锁频。每组内部采用相反 batch 顺序复测。数据是该模型、GPU、runtime 和提示集合的测量，不是跨模型或整服务吞吐结论。",
             "- 之前人工 hot/evicted cold 的报告仅解释内核端点；部署结论以本报告的自然缓存结果为准。", "",
             "隔离环境准备完成后，以同一 GPU 分别运行两个 arm：", "", "```bash",
             'CUDA_VISIBLE_DEVICES="${GPU_ID}" \\',
             'MXFP8_LIBRARY_PATH="${KERNEL_ROOT}/build/mxfp8/mxfp8_torch.so" \\',
             'PYTHONPATH="${AUDIT_ROOT}/site:${KERNEL_ROOT}/python:${KERNEL_ROOT}/benchmarks" \\',
             '"${RUNTIME_PYTHON}" \\',
             "  benchmarks/benchmark_mxfp8_dual_inference.py --arm single \\",
             '  --model "${MODEL_DIR}" --prompts "${PROMPTS_FILE}" \\',
             '  --output "${AUDIT_ROOT}/new-single" \\',
             "  --repeats 2 --output-tokens 32", "```", "",
             "将 `--arm single` 改为 `dual` 并使用不同输出目录即可测另一组。完整前向验证添加环境变量 `MX8_INFERENCE_EVENTS=0` 和参数 `--extra-context 3000`。", "",
             "复现命令中的变量按本地环境设置：`KERNEL_ROOT` 为内核仓库，`RUNTIME_PYTHON` 为 runtime 的 Python 环境，`MODEL_DIR` 为模型目录，`AUDIT_ROOT` 为本地审计目录，`PROMPTS_FILE` 为提示 JSON，`GPU_ID` 为测试设备编号。", ""]
    return "\n".join(text)


def plot(artifact, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[r for r in artifact["points"] if r["context"]==256]
    batches=[r["batch"] for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for axis,kind,title in [(axes[0,0],"qkvz","QKVZ, all 24 layers"),(axes[0,1],"gateup","Gate/up, all 32 layers")]:
        for role,color in [("single","#147d92"),("dual","#dc6533")]:
            axis.plot(batches,[r[f"{role}_steady_{kind}_gemm_us"] for r in rows],label=role+" steady",color=color,lw=2)
            axis.plot(batches,[r[f"{role}_first_{kind}_gemm_us"] for r in rows],label=role+" first",color=color,lw=1,ls=":",alpha=.6)
        axis.set(title=title,ylabel="GEMM interval (us), quantization excluded")
        axis.legend(frameon=False)
    for kind,color in [("qkvz","#7461a3"),("gateup","#388547")]:
        axes[1,0].plot(batches,[r[f"steady_{kind}_gemm_us_ratio"] for r in rows],color=color,lw=2,label=kind+" steady")
        axes[1,0].plot(batches,[r[f"first_{kind}_gemm_us_ratio"] for r in rows],color=color,ls=":",alpha=.6,label=kind+" first")
    axes[1,0].set(title="GEMM-only overhead inside real model inference",ylabel="Dual / single latency (lower is better)")
    axes[1,0].axhline(1,color="#555",lw=1,ls="--")
    axes[1,0].legend(frameon=False)
    axes[1,1].plot(batches,[r["steady_backbone_ratio"] for r in rows],label="ISL256, all batches",lw=2,color="#2761ad")
    long=[r for r in artifact["points"] if r["context"]==3000]
    axes[1,1].plot([r["batch"] for r in long],[r["steady_backbone_ratio"] for r in long],label="ISL3000, representative batches",lw=2,marker="o",color="#9a6231")
    axes[1,1].axhline(1,color="#555",lw=1,ls="--")
    axes[1,1].set(title="Secondary check: unfused backbone, no inner events",ylabel="Dual / single forward latency")
    axes[1,1].legend(frameon=False)
    for axis in axes.flat:
        axis.set(xlabel="Requested batch size",xlim=(1,128))
        axis.grid(alpha=.2)
    fig.suptitle("Qwen3.5-4B / RTX 5090 / natural inference cache states",fontsize=15)
    fig.savefig(destination)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--plot",action="store_true")
    args=ap.parse_args()
    datasets={"single":collect(args.root/"single-v2",True),"dual":collect(args.root/"dual",True),
              "single_forward":collect(args.root/"single-forward",False),
              "dual_forward":collect(args.root/"dual-forward",False)}
    required={(256,b) for b in range(1,129)}
    assert set(datasets["single"]) == set(datasets["dual"]) == required
    assert set(datasets["single_forward"]) == set(datasets["dual_forward"])
    assert required <= set(datasets["single_forward"])
    records=[]
    for context,batch in sorted(datasets["single_forward"]):
        point={"context":context,"batch":batch}
        for phase in ("first","steady"):
            for role in ("single","dual"):
                repeats=datasets[role+"_forward"][(context,batch)][phase]
                point["physical_m"]=next(iter(repeats.values()))["physical_m"]
                point[f"{role}_{phase}_backbone_ms"]=average(repeats,"backbone_ms")
                point[f"{role}_{phase}_steps"]=sum(r["steps"] for r in repeats.values())
                if (context,batch) in datasets[role]:
                    point[f"{role}_{phase}_instrumented_backbone_ms"]=average(datasets[role][(context,batch)][phase],"instrumented_backbone_ms")
                    for kind in ("qkvz","gateup"):
                        for metric in ("quant_us","gemm_us","total_us"):
                            point[f"{role}_{phase}_{kind}_{metric}"]=average(datasets[role][(context,batch)][phase],f"{kind}_{metric}")
            point[f"{phase}_backbone_ratio"]=point[f"dual_{phase}_backbone_ms"]/point[f"single_{phase}_backbone_ms"]
            if (context,batch) in required:
                for kind in ("qkvz","gateup"):
                    for metric in ("gemm_us","total_us"):
                        point[f"{phase}_{kind}_{metric}_ratio"]=point[f"dual_{phase}_{kind}_{metric}"]/point[f"single_{phase}_{kind}_{metric}"]
        records.append(point)
    summary={}
    for context in sorted({r["context"] for r in records}):
        group=[r for r in records if r["context"]==context]
        summary[str(context)]={}
        for phase in ("first","steady"):
            for metric in ("backbone","qkvz_total_us","gateup_total_us","qkvz_gemm_us","gateup_gemm_us"):
                key=f"{phase}_{metric}_ratio"
                values=[r[key] for r in group if key in r]
                if values:
                    summary[str(context)][key]={"geomean":math.exp(stats.fmean(math.log(v) for v in values)),
                        "min":min(values),"max":max(values),"dual_faster_points":sum(v<1 for v in values),"points":len(values)}
    artifact={"schema":1,"primary_boundary":"GEMM-only, quantization excluded; future activation+quantization fusion not implemented", "date":"2026-10-09","comparison":"native dual / native single, ordinary launch (PDL off)",
              "cache_method":"Natural full-model execution; first full-batch decode after prefill and steady decode. No synthetic cache manipulation.",
              "aggregation":"Within-repeat median of step-level layer means; arithmetic mean of two forward/reverse sweeps; geometric mean of per-batch ratios.",
              "steady_selection":"Skip first 3 and last 2 full-batch decode steps; all 56 target projections verified in each instrumented step.",
              "identities":{key:identity(args.root/name) for key,name in [("single","single-v2"),("dual","dual"),("single_forward","single-forward"),("dual_forward","dual-forward")]},
              "summary":summary,"points":records,
              "repeat_summary":{role:{f"ISL{context}:BS{batch}":phases
                                      for (context,batch),phases in points.items()}
                                for role,points in datasets.items()},
              "model_identity":json.loads((args.root/"model_identity.json").read_text()),
              "isolation":json.loads((args.root/"isolation.json").read_text())}
    previous=Path(__file__).with_name("results")/"mxfp8_dual_bs1_128.json"
    if previous.exists():
        p=json.loads(previous.read_text())
        artifact["gpu"]=p["gpu"]
        artifact["environment"]={**p["environment"],"date_utc":"2026-10-09","benchmark":"natural full-model inference"}
    artifact["batch_ranges"]={}
    artifact["validation"]={"requested_batches":list(range(1,129)),
                            "actual_m":sorted({r["physical_m"] for r in records if r["context"]==256}),
                            "gateup_layers":32,"qkvz_layers":24,
                            "minimum_instrumented_steady_steps_per_repeat":min(
                                v["steps"] for role in ("single","dual")
                                for phases in datasets[role].values() for v in phases["steady"].values())}
    for lower,upper in [(1,32),(33,64),(65,128)]:
        group=[r for r in records if r["context"]==256 and lower<=r["batch"]<=upper]
        artifact["batch_ranges"][f"{lower}-{upper}"]={
            metric:math.exp(stats.fmean(math.log(r[metric]) for r in group))
            for metric in ("steady_qkvz_gemm_us_ratio","steady_gateup_gemm_us_ratio","steady_qkvz_total_us_ratio","steady_gateup_total_us_ratio","steady_backbone_ratio")}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    artifact = redact_local_metadata(artifact)
    args.output.with_suffix(".json").write_text(json.dumps(artifact,indent=2,allow_nan=False)+"\n")
    args.output.with_suffix(".md").write_text(render_markdown(artifact,args.output.name))
    keys=list(dict.fromkeys(k for r in records for k in r))
    with args.output.with_suffix(".csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)
    if args.plot:
        plot(artifact,args.output.with_suffix(".svg"))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
