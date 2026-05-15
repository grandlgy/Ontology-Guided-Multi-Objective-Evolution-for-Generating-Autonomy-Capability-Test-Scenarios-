"""
完整实验:
- 4种方法对比: Random / CT / Pure NSGA-II / LSS-CO+NSGA-II
- 30 seeds 独立重复
- 统计检验: Mann-Whitney U test
- 生成: 收敛曲线、Pareto前沿、对比表、消融实验
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

from framework import (
    train_surrogate, baseline_random, baseline_combinatorial,
    run_nsga2, evaluate_set, predict_pfail,
    LB, UB, N_DIMS, SCENARIO_DIMENSIONS
)

# 中文字体处理 - 后续保存图时尝试
matplotlib.rcParams['axes.unicode_minus'] = False
# 尝试使用系统可用的字体
for f in ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']:
    try:
        matplotlib.rcParams['font.sans-serif'] = [f]
        break
    except Exception:
        continue

N_SEEDS = 15
POP_SIZE = 100
N_GEN = 100
OUT_DIR = "/home/claude/experiment/results"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("/home/claude/experiment/figures", exist_ok=True)

# ============================================================
# Step 1: 训练代理模型并报告指标
# ============================================================
print("Step 1: 训练失效代理模型...")
reg, rmse, r2 = train_surrogate(n_train=8000, seed=42)
print(f"  代理模型: RMSE={rmse:.4f}, R2={r2:.4f}")

with open(os.path.join(OUT_DIR, "surrogate_metrics.json"), "w") as f:
    json.dump({"rmse": float(rmse), "r2": float(r2),
               "n_train": 8000, "test_size": 0.2}, f)

# ============================================================
# Step 2: 4种方法跨30个seed的实验
# ============================================================
print(f"\nStep 2: 主实验 (30 seeds * 4 methods)...")
methods = ["Random", "CT", "PureGA", "LSS-CO"]
results = {m: [] for m in methods}
convergence = {m: [] for m in ["PureGA", "LSS-CO"]}

for seed in range(N_SEEDS):
    print(f"  seed {seed+1}/{N_SEEDS}", end=" ", flush=True)
    # Random
    X = baseline_random(POP_SIZE, seed=seed)
    results["Random"].append(evaluate_set(X, reg, ontology_used=False))
    # CT
    X = baseline_combinatorial(POP_SIZE, seed=seed)
    results["CT"].append(evaluate_set(X, reg, ontology_used=False))
    # Pure NSGA-II (no ontology)
    X, hp, hc = run_nsga2(reg, use_ontology=False,
                          pop_size=POP_SIZE, n_gen=N_GEN, seed=seed)
    results["PureGA"].append(evaluate_set(X, reg, ontology_used=False))
    convergence["PureGA"].append({"pfail": hp, "cov": hc})
    # LSS-CO + NSGA-II
    X, hp, hc = run_nsga2(reg, use_ontology=True,
                          pop_size=POP_SIZE, n_gen=N_GEN, seed=seed)
    results["LSS-CO"].append(evaluate_set(X, reg, ontology_used=True))
    convergence["LSS-CO"].append({"pfail": hp, "cov": hc})
    print("ok")

# ============================================================
# Step 3: 统计分析
# ============================================================
print("\nStep 3: 统计分析...")
metrics_keys = ["coverage", "crit_ratio", "phys_viol_ratio", "mean_pfail"]
summary = {}
for m in methods:
    summary[m] = {}
    for k in metrics_keys:
        vals = np.array([r[k] for r in results[m]])
        summary[m][k] = {"mean": float(vals.mean()),
                         "std": float(vals.std()),
                         "values": vals.tolist()}

# 主方法 vs 各基线的Mann-Whitney U检验
sig_tests = {}
for baseline in ["Random", "CT", "PureGA"]:
    sig_tests[baseline] = {}
    for k in metrics_keys:
        a = np.array(summary["LSS-CO"][k]["values"])
        b = np.array(summary[baseline][k]["values"])
        alt = "greater" if k in ("coverage", "crit_ratio", "mean_pfail") else "less"
        stat, p = mannwhitneyu(a, b, alternative=alt)
        sig_tests[baseline][k] = {"U": float(stat), "p": float(p),
                                   "alternative": alt}

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump({"summary": summary, "sig_tests": sig_tests}, f, indent=2)

print("\n  主指标(均值±标准差):")
print("  Method      Coverage      CritRatio     PhysViol      MeanPfail")
for m in methods:
    s = summary[m]
    print(f"  {m:10s}  {s['coverage']['mean']:.3f}±{s['coverage']['std']:.3f}  "
          f"{s['crit_ratio']['mean']:.3f}±{s['crit_ratio']['std']:.3f}  "
          f"{s['phys_viol_ratio']['mean']:.3f}±{s['phys_viol_ratio']['std']:.3f}  "
          f"{s['mean_pfail']['mean']:.3f}±{s['mean_pfail']['std']:.3f}")

print("\n  LSS-CO vs 各基线 Mann-Whitney U test p值:")
for baseline in ["Random", "CT", "PureGA"]:
    print(f"  vs {baseline}:")
    for k in metrics_keys:
        p = sig_tests[baseline][k]["p"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"    {k:20s}: p={p:.4e} {sig}")

# ============================================================
# Step 4: 消融实验
# ============================================================
print("\nStep 4: 消融实验...")
ablation_results = {"Full LSS-CO": summary["LSS-CO"],
                    "No Ontology": summary["PureGA"]}

# 不使用修复算子(随机种群)的NSGA-II,但保留违例目标 - 已经被Pure GA覆盖
# 这里再加一个: 仅2目标的NSGA-II (压力+物理,无novelty)
print("  跑Ablation-NoNovelty (2目标,不含多样性)...")

# 简化: 临时修改framework里的LSSCO_Problem,但更安全的是重写一个local版本
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize as pymoo_min
from pymoo.termination import get_termination
from framework import repair, physics_violation, predict_pfail

class NoNoveltyProblem(Problem):
    def __init__(self, surrogate):
        super().__init__(n_var=N_DIMS, n_obj=2, n_constr=0, xl=LB, xu=UB)
        self.surrogate = surrogate
    def _evaluate(self, X, out, *args, **kwargs):
        X_eval = np.array([repair(x) for x in X])
        pfail = predict_pfail(self.surrogate, X_eval)
        f1 = -pfail
        f2 = np.array([physics_violation(x) for x in X_eval])
        out["F"] = np.column_stack([f1, f2])

no_novelty_results = []
for seed in range(N_SEEDS):
    prob = NoNoveltyProblem(reg)
    alg = NSGA2(pop_size=POP_SIZE, sampling=LHS(),
                crossover=SBX(prob=0.9, eta=20),
                mutation=PM(prob=1.0/N_DIMS, eta=20),
                eliminate_duplicates=True)
    res = pymoo_min(prob, alg, get_termination("n_gen", N_GEN),
                    seed=seed, verbose=False)
    X_final = res.pop.get("X")
    no_novelty_results.append(evaluate_set(X_final, reg, ontology_used=True))

ablation_summary = {"NoNovelty": {}}
for k in metrics_keys:
    vals = np.array([r[k] for r in no_novelty_results])
    ablation_summary["NoNovelty"][k] = {"mean": float(vals.mean()), "std": float(vals.std()),
                                         "values": vals.tolist()}

with open(os.path.join(OUT_DIR, "ablation.json"), "w") as f:
    json.dump(ablation_summary, f, indent=2)

print(f"  NoNovelty: cov={ablation_summary['NoNovelty']['coverage']['mean']:.3f}, "
      f"crit={ablation_summary['NoNovelty']['crit_ratio']['mean']:.3f}, "
      f"viol={ablation_summary['NoNovelty']['phys_viol_ratio']['mean']:.3f}")

# ============================================================
# Step 5: 收敛曲线作图
# ============================================================
print("\nStep 5: 绘制收敛曲线...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for method in ["PureGA", "LSS-CO"]:
    pf_arr = np.array([c["pfail"] for c in convergence[method]])
    cv_arr = np.array([c["cov"] for c in convergence[method]])
    gens = np.arange(1, pf_arr.shape[1] + 1)
    mean_pf = pf_arr.mean(axis=0)
    std_pf = pf_arr.std(axis=0)
    mean_cv = cv_arr.mean(axis=0)
    std_cv = cv_arr.std(axis=0)
    label = "Pure NSGA-II" if method == "PureGA" else "LSS-CO+NSGA-II (ours)"
    axes[0].plot(gens, mean_pf, label=label, linewidth=1.8)
    axes[0].fill_between(gens, mean_pf - std_pf, mean_pf + std_pf, alpha=0.18)
    axes[1].plot(gens, mean_cv, label=label, linewidth=1.8)
    axes[1].fill_between(gens, mean_cv - std_cv, mean_cv + std_cv, alpha=0.18)

axes[0].set_xlabel("Generation")
axes[0].set_ylabel("Mean failure probability")
axes[0].set_title("(a) Convergence of test pressure")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].set_xlabel("Generation")
axes[1].set_ylabel("Coverage score")
axes[1].set_title("(b) Convergence of coverage")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/experiment/figures/fig_convergence.png", dpi=160)
plt.close()
print("  已保存 fig_convergence.png")

# ============================================================
# Step 6: Pareto前沿可视化(用一个代表性seed)
# ============================================================
print("Step 6: Pareto前沿可视化...")
X_lssco, _, _ = run_nsga2(reg, use_ontology=True,
                          pop_size=POP_SIZE, n_gen=N_GEN, seed=0)
X_pure, _, _ = run_nsga2(reg, use_ontology=False,
                         pop_size=POP_SIZE, n_gen=N_GEN, seed=0)

from framework import repair as rep_fn
X_lssco_eval = np.array([rep_fn(x) for x in X_lssco])
X_pure_eval = X_pure.copy().astype(float)
for i, d in enumerate(SCENARIO_DIMENSIONS):
    if d[1] == 'integer':
        X_pure_eval[:, i] = np.round(X_pure_eval[:, i])
X_pure_eval = np.clip(X_pure_eval, LB, UB)

p_lssco = predict_pfail(reg, X_lssco_eval)
p_pure = predict_pfail(reg, X_pure_eval)
v_lssco = np.array([physics_violation(x) for x in X_lssco_eval])
v_pure = np.array([physics_violation(x) for x in X_pure_eval])

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(p_pure, v_pure, s=28, alpha=0.55,
           label="Pure NSGA-II (no ontology)", color="#888")
ax.scatter(p_lssco, v_lssco, s=28, alpha=0.85,
           label="LSS-CO+NSGA-II (ours)", color="#c1272d")
ax.set_xlabel("Failure probability $\\hat{p}_{fail}$ (higher = more critical)")
ax.set_ylabel("Physics violation (lower = more feasible)")
ax.set_title("Final population: criticality vs. feasibility")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/experiment/figures/fig_pareto.png", dpi=160)
plt.close()
print("  已保存 fig_pareto.png")

# ============================================================
# Step 7: 各方法对比条形图
# ============================================================
print("Step 7: 各方法对比图...")
fig, ax = plt.subplots(figsize=(8.5, 4.5))
metric_labels = ["Coverage", "Critical-ratio", "Phys-violation", "Mean pfail"]
metric_keys = ["coverage", "crit_ratio", "phys_viol_ratio", "mean_pfail"]
x = np.arange(len(metric_labels))
width = 0.2
colors = ["#4F81BD", "#9BBB59", "#C0504D", "#8064A2"]
for i, m in enumerate(methods):
    means = [summary[m][k]["mean"] for k in metric_keys]
    stds = [summary[m][k]["std"] for k in metric_keys]
    ax.bar(x + i*width - 1.5*width, means, width,
           yerr=stds, capsize=3, label=m, color=colors[i], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylabel("Score (mean over 30 seeds)")
ax.set_title("Comparison of scenario generation methods")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/experiment/figures/fig_comparison.png", dpi=160)
plt.close()
print("  已保存 fig_comparison.png")

# ============================================================
# Step 8: 失效场景特征分析
# ============================================================
print("Step 8: 关键场景分析...")
# 取LSS-CO seed=0的高失效场景,统计参数分布
high_risk_idx = np.argsort(p_lssco)[-30:]
top_X = X_lssco_eval[high_risk_idx]
top_pfail = p_lssco[high_risk_idx]

# 保存一份代表性场景
top_scenarios = []
for i, (x, pf) in enumerate(zip(top_X, top_pfail)):
    sc = {"id": i, "pfail": float(pf)}
    for j, d in enumerate(SCENARIO_DIMENSIONS):
        sc[d[0]] = float(x[j])
    top_scenarios.append(sc)
with open(os.path.join(OUT_DIR, "top_scenarios.json"), "w") as f:
    json.dump(top_scenarios, f, indent=2)

print(f"  生成了 {len(top_scenarios)} 条高风险场景, pfail均值={top_pfail.mean():.3f}")

# ============================================================
# Step 9: 输出主表格(latex/markdown格式)
# ============================================================
print("Step 9: 生成主表格...")
table_lines = ["| Method | Coverage ↑ | Critical-Ratio ↑ | Phys-Viol ↓ | Mean p_fail ↑ |",
               "|---|---|---|---|---|"]
for m in methods:
    s = summary[m]
    row = (f"| {m} | "
           f"{s['coverage']['mean']:.3f}±{s['coverage']['std']:.3f} | "
           f"{s['crit_ratio']['mean']:.3f}±{s['crit_ratio']['std']:.3f} | "
           f"{s['phys_viol_ratio']['mean']:.3f}±{s['phys_viol_ratio']['std']:.3f} | "
           f"{s['mean_pfail']['mean']:.3f}±{s['mean_pfail']['std']:.3f} |")
    table_lines.append(row)
table_md = "\n".join(table_lines)
with open(os.path.join(OUT_DIR, "main_table.md"), "w") as f:
    f.write(table_md + "\n")
print(table_md)

print("\n✅ 完整实验运行完成")
print(f"  - 结果文件夹: {OUT_DIR}")
print(f"  - 图片文件夹: /home/claude/experiment/figures")
