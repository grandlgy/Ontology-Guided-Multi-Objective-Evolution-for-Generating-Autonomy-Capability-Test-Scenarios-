"""
SITL reproduction experiment.

1. Run the three top critical scenarios (C1, C2, C3) selected by LSS-CO+NSGA-II
   through the lightweight simulator, multiple seeds per scenario.
2. Validate the XGBoost surrogate against the lightweight simulator on a
   randomly drawn validation set of scenarios.
3. Produce trajectory plots and a validation table.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from sitl_simulator import simulate_scenario, batch_simulate
from framework import (
    train_surrogate, baseline_random, predict_pfail,
    SCENARIO_DIMENSIONS, LB, UB, repair, physics_violation
)

OUT = "/home/claude/experiment/results_sitl"
FIG = "/home/claude/experiment/figures"
os.makedirs(OUT, exist_ok=True)

# -----------------------------------------------------------
# 1. Define the three critical scenarios (from previous run)
# -----------------------------------------------------------
def make_scenario(values):
    return {SCENARIO_DIMENSIONS[i][0]: values[i] for i in range(len(values))}

C1 = make_scenario([13.92, 226.62, 0, 1, 0.76, 4.28, 1, -4.82, 0.97,
                    3, 34.99, 19.56, 0.99, 482.30, 0.34, 16.59, 915.24])
C2 = make_scenario([13.92, 226.60, 0, 1, 0.76, 4.17, 0, -21.99, 1.00,
                    3, 35.00, 19.57, 0.99, 482.15, 0.02, 16.63, 1081.14])
C3 = make_scenario([13.77, 338.47, 0, 1, 0.76, 4.28, 1, -22.68, 0.88,
                    3, 34.99, 19.37, 0.99, 417.08, 0.35, 16.59, 914.29])

# Repair them to ensure ontology validity
def repair_dict(d):
    arr = np.array([d[k] for k, *_ in SCENARIO_DIMENSIONS])
    arr = repair(arr)
    return {SCENARIO_DIMENSIONS[i][0]: float(arr[i]) for i in range(len(arr))}

C1 = repair_dict(C1)
C2 = repair_dict(C2)
C3 = repair_dict(C3)

# Also prepare a benign baseline scenario for visual contrast
BENIGN = {
    'env_wind_speed': 2.0, 'env_visibility': 4500, 'env_illumination': 3,
    'env_terrain': 0, 'env_em_interference': 0.05, 'env_precipitation': 0.0,
    'tgt_platform_type': 0, 'tgt_rcs_dbsm': -10, 'tgt_obstacle_density': 0.1,
    'mis_type': 0, 'mis_altitude': 80, 'mis_velocity': 8.0,
    'dist_gps_loss': 0.05, 'dist_comm_jitter': 50, 'dist_payload_change': 0.0,
    'dist_battery_level': 85, 'scn_init_distance': 200,
}

print("=" * 60)
print("Step 1: Critical scenarios reproduction (20 seeds each)")
print("=" * 60)
scenarios = [BENIGN, C1, C2, C3]
labels = ["Benign", "C1", "C2", "C3"]
N_RUNS = 20

sitl_results = []
for name, sc in zip(labels, scenarios):
    runs = []
    for r in range(N_RUNS):
        log = simulate_scenario(sc, seed=1000 + r, return_log=False)
        runs.append({'success': log.success,
                     'time': log.total_time,
                     'reason': log.failure_reason})
    fails = sum(1 for r in runs if not r['success'])
    rate = fails / N_RUNS
    reasons = [r['reason'] for r in runs if not r['success']]
    from collections import Counter
    rc = Counter(reasons)
    print(f"  {name}: failure_rate = {rate:.3f}  reasons = {dict(rc)}")
    sitl_results.append({'name': name, 'failure_rate': rate,
                         'reasons': dict(rc), 'runs': runs})

with open(os.path.join(OUT, "critical_scenarios_sitl.json"), "w") as f:
    json.dump(sitl_results, f, indent=2)

# -----------------------------------------------------------
# 2. Surrogate validation: surrogate prediction vs SITL outcome
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Surrogate model validation against the simulator")
print("=" * 60)

# Train surrogate (same as in main experiment)
print("  Training surrogate...")
clf, rmse, r2 = train_surrogate(n_train=8000, seed=42)
print(f"  Surrogate: RMSE = {rmse:.4f},  R2 = {r2:.4f}")

# Draw 200 random validation scenarios
print("  Drawing 200 validation scenarios for SITL comparison...")
rng = np.random.default_rng(0)
X_val = rng.uniform(LB, UB, size=(200, len(LB)))
X_val = np.array([repair(x) for x in X_val])

surrogate_preds = predict_pfail(clf, X_val)

print("  Running SITL on each scenario (5 seeds each)...")
sitl_failure_rates = []
for i, x in enumerate(X_val):
    sc = {SCENARIO_DIMENSIONS[j][0]: float(x[j]) for j in range(len(x))}
    fails = 0
    for r in range(5):
        log = simulate_scenario(sc, seed=2000 + i*10 + r, return_log=False)
        if not log.success:
            fails += 1
    sitl_failure_rates.append(fails / 5)
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/200 done")

sitl_failure_rates = np.array(sitl_failure_rates)
# Compare surrogate vs SITL with Pearson, Spearman, ROC-AUC at threshold 0.5
from scipy.stats import pearsonr, spearmanr
pearson, _ = pearsonr(surrogate_preds, sitl_failure_rates)
spearman, _ = spearmanr(surrogate_preds, sitl_failure_rates)
# For AUC: treat SITL outcome >= 0.5 as "failure" and surrogate as score
sitl_binary = (sitl_failure_rates >= 0.5).astype(int)
if sitl_binary.sum() > 0 and sitl_binary.sum() < len(sitl_binary):
    auc = roc_auc_score(sitl_binary, surrogate_preds)
else:
    auc = float('nan')

print(f"\n  Surrogate vs SITL on 200 random scenarios:")
print(f"    Pearson  r = {pearson:.4f}")
print(f"    Spearman r = {spearman:.4f}")
print(f"    ROC-AUC    = {auc:.4f}  (SITL fail >= 50% as positive)")

with open(os.path.join(OUT, "surrogate_validation.json"), "w") as f:
    json.dump({'pearson': float(pearson),
               'spearman': float(spearman),
               'roc_auc': float(auc) if not np.isnan(auc) else None,
               'n_validation': 200,
               'runs_per_scenario': 5,
               'mean_surrogate': float(surrogate_preds.mean()),
               'mean_sitl': float(sitl_failure_rates.mean())}, f, indent=2)

# -----------------------------------------------------------
# 3. Trajectory plots for critical scenarios
# -----------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Trajectory plots for critical scenarios")
print("=" * 60)

fig = plt.figure(figsize=(13, 4.6))
for idx, (name, sc) in enumerate(zip(["Benign baseline", "C1", "C2", "C3"], scenarios)):
    ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
    # Run 3 example flights and plot
    final_states = []
    for r in range(3):
        log = simulate_scenario(sc, seed=300 + r, return_log=True)
        if len(log.pos) > 0:
            arr = np.array(log.pos)
            color = 'tab:green' if log.success else 'tab:red'
            ax.plot(arr[:,0], arr[:,1], arr[:,2], color=color, alpha=0.7, linewidth=1.3)
            ax.scatter(arr[-1,0], arr[-1,1], arr[-1,2], color=color, s=22)
            final_states.append(log.failure_reason)
    ax.set_title(f"{name}\noutcome: {final_states[0] if final_states else 'n/a'}",
                 fontsize=9.5)
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_sitl_trajectories.png"), dpi=170,
            bbox_inches='tight')
plt.close()
print(f"  Saved fig_sitl_trajectories.png")

# Time series plot for one critical scenario
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, sc, name in zip(axes, [BENIGN, C1, C3], ["Benign baseline", "C1", "C3"]):
    log = simulate_scenario(sc, seed=500, return_log=True)
    if len(log.t) > 0:
        ts = np.array(log.t)
        gps_err = np.array(log.gps_err)
        bat = np.array(log.battery)
        ax2 = ax.twinx()
        ln1 = ax.plot(ts, gps_err, 'b-', label='GPS bias norm (m)', linewidth=1.4)
        ln2 = ax2.plot(ts, bat, 'r-', label='Battery (%)', linewidth=1.4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("GPS bias norm (m)", color='b')
        ax2.set_ylabel("Battery (%)", color='r')
        ax.set_title(f"{name}: {log.failure_reason}", fontsize=10)
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_sitl_timeseries.png"), dpi=170,
            bbox_inches='tight')
plt.close()
print("  Saved fig_sitl_timeseries.png")

# Surrogate vs SITL correlation plot
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(surrogate_preds, sitl_failure_rates, s=18, alpha=0.55,
           color='steelblue', edgecolors='navy', linewidths=0.4)
# Diagonal
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel("Surrogate $\\hat{p}_{fail}$")
ax.set_ylabel("SITL empirical failure rate (5 runs)")
ax.set_title(f"Surrogate vs SITL  (n=200, Pearson r={pearson:.3f}, AUC={auc:.3f})")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_surrogate_vs_sitl.png"), dpi=170,
            bbox_inches='tight')
plt.close()
print("  Saved fig_surrogate_vs_sitl.png")

print("\nAll SITL artifacts saved to:", OUT)
print("Figures saved to:", FIG)
