# LSS-CO: Scenario Generation for Low-Slow-Small UAV Autonomy Testing

This repository contains the source code, ontology file, and raw experimental
artifacts that accompany the paper:

> **Ontology-Guided Multi-Objective Evolution for Generating Autonomy Capability Test Scenarios of Low-Slow-Small UAVs**
> *Submitted to MDPI Drones, 2026*

## What is in here

```
.
├── code/
│   ├── framework.py           # LSS-CO + NSGA-II + baselines + metrics
│   ├── sitl_simulator.py      # Lightweight 6-DoF UAV testbed for SITL reproduction
│   ├── run_experiment.py      # Main comparison: 4 methods x 15 seeds
│   ├── run_sitl.py            # Reproduces C1/C2/C3 and surrogate validation
│   └── draw_diagrams.py       # Generates framework and ontology figures
├── ontology/
│   └── LSS-CO.owl             # OWL2 DL skeleton of the scenario ontology
├── results/                   # Raw experiment outputs (JSON, all figures)
├── figures/                   # Reproduced figures from the paper
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

### Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11, pymoo 0.6.1, XGBoost 3.2.0, scikit-learn 1.5,
matplotlib 3.8, numpy 1.26.

### Reproduce the main comparison

```bash
cd code
python run_experiment.py
```

Runtime is around 35 minutes on a single CPU (Intel Xeon Gold 6326).
Outputs:

- `results/summary.json`: per-method aggregate metrics across 15 seeds.
- `results/ablation.json`: ablation result for the no-novelty variant.
- `results/top_scenarios.json`: top 30 critical scenarios (LSS-CO+NSGA-II, seed 0).
- `figures/fig_convergence.png`: convergence curves of pressure and coverage.
- `figures/fig_pareto.png`: final populations in (criticality, violation) plane.
- `figures/fig_comparison.png`: per-method bar plot comparison.

### Reproduce the SITL experiment

```bash
cd code
python run_sitl.py
```

Runtime is around 8 minutes. Outputs:

- `results_sitl/critical_scenarios_sitl.json`: failure rate of C1, C2, C3.
- `results_sitl/surrogate_validation.json`: surrogate-vs-simulator correlation.
- `figures/fig_sitl_trajectories.png`: 3D trajectories.
- `figures/fig_sitl_timeseries.png`: GPS bias and battery time series.
- `figures/fig_surrogate_vs_sitl.png`: cross-check scatter plot.

## Result highlights

| Method | Coverage ↑ | Critical-ratio ↑ | Phys-violation ↓ | Mean p_fail ↑ |
|---|---|---|---|---|
| Random | 0.723 ± 0.012 | 0.155 ± 0.041 | 0.480 ± 0.047 | 0.395 ± 0.013 |
| Combinatorial testing | 0.394 ± 0.004 | 0.292 ± 0.033 | 0.538 ± 0.030 | 0.428 ± 0.009 |
| NSGA-II (no ontology) | 0.209 ± 0.023 | 0.912 ± 0.029 | 0.341 ± 0.045 | 0.752 ± 0.017 |
| **LSS-CO + NSGA-II (ours)** | 0.225 ± 0.027 | **0.849 ± 0.055** | **0.287 ± 0.073** | 0.714 ± 0.030 |

SITL reproduction of the three top critical scenarios: 60/60 failed runs
(20 seeds × 3 scenarios), compared to 0/20 for a benign baseline.

Mann-Whitney U test against random and combinatorial baselines reaches
*p* < 10⁻⁵ on the two pressure-related metrics. Against plain NSGA-II,
the improvement on physics violation is significant at *p* < 0.05.

## How to extend

### Add a new capability dimension

Edit `framework.py`:

1. Append a tuple `(name, type, lower, upper)` to `SCENARIO_DIMENSIONS`.
2. Add the corresponding contribution to `synthesize_failure_score`
   if the new dimension affects failure probability.
3. Optionally extend `physics_violation` with a cross-domain axiom that
   the new dimension participates in.

### Replace the surrogate label source

The framework treats the surrogate as a black box. To use real flight-log
labels, prepare a CSV with one row per scenario (17 columns matching the
order in `SCENARIO_DIMENSIONS`) and a binary or continuous failure label.
Replace `synthesize_failure_score` with a loader and re-run `train_surrogate`.

### Plug a different simulator

`sitl_simulator.simulate_scenario(sc, seed)` takes a dict with the 17 keys
and returns a `SimLog`. Any simulator that exposes the same interface can
be substituted: AirSim with PX4 SITL, Gazebo, jMAVSim, or a hardware-in-the-loop
rig with an autopilot board.

## License

MIT. See `LICENSE`.

## Citation

If you use this code or the LSS-CO ontology, please cite the paper:

```bibtex
@article{lssco2026,
  title  = {Ontology-Guided Multi-Objective Evolution for Generating
            Autonomy Capability Test Scenarios of Low-Slow-Small UAVs},
  author = {Author 1 and Author 2},
  journal= {Drones},
  year   = {2026},
  publisher = {MDPI},
}
```

## Contact

For questions and collaboration: corresponding.author@example.edu
