# Offshore Wind O&M — Reliability-Based Resource Design and Opportunistic Routing

> **Paper:** *Reliability-Based Maintenance Resource Design and Opportunistic Routing Optimization for Offshore Wind Farms under Weather Uncertainty*

---

## Overview

This repository contains the full Python + R implementation for the numerical experiments and publication figures of the paper. The model formulates a **two-stage stochastic mixed-integer nonlinear program (MINLP)** that jointly optimizes:

- **First stage:** O&M base selection, vessel fleet deployment, technician skill composition
- **Second stage:** Rolling-horizon route–maintenance pattern selection under weather uncertainty

Key modelling features:
- Nonlinear Weibull effective-age degradation per component
- Reliability-based expected production coefficient
- Imperfect maintenance levels L1 / L2 / L3 with technician qualification constraints
- Opportunistic multi-turbine bundling with shared routing cost
- No-action gate to prevent heuristically overestimated maintenance execution

---

## Repository Structure

```
offshore-wind-om/
│
├── run_all.py              ← Master runner (start here)
│
├── src/
│   ├── parameters.py       ← All model parameters and constants
│   ├── reliability.py      ← Weibull reliability, degradation, production coefficient
│   ├── patterns.py         ← Route–maintenance pattern generation and scoring
│   ├── simulation.py       ← Rolling-horizon simulation engine
│   └── experiments.py      ← Experiment runner (configs, policies, sensitivity)
│
├── visualization.R         ← Publication-quality figures (ggplot2 / patchwork)
│
├── results/                ← Auto-generated: experiment_results.json
└── figures/                ← Auto-generated: fig1_*.png … fig7_*.png
```

---

## Quickstart

### 1. Requirements

**Python ≥ 3.9**
```bash
pip install numpy
```

**R ≥ 4.0** *(optional — for publication-quality figures)*
```r
install.packages(c("ggplot2", "patchwork", "dplyr", "tidyr",
                   "scales", "RColorBrewer", "jsonlite"))
```

### 2. Run everything

```bash
python run_all.py
```

This single command:
1. Validates Python dependencies
2. Runs all numerical experiments → `results/experiment_results.json`
3. Generates all 7 figures (R if available, else matplotlib fallback) → `figures/`
4. Prints a result summary to the console

### 3. Partial runs

```bash
# Experiments only (no figures)
python run_all.py --no-figures

# Re-generate figures from existing results
python run_all.py --figs-only

# Custom output directories
python run_all.py --results-dir my_results --figures-dir my_figures
```

### 4. Run individual modules

```bash
# From the repository root:
python src/experiments.py          # run experiments and save JSON
Rscript visualization.R            # generate figures (requires results/ JSON)
```

---

## Experiments

| # | Experiment | Output | Table / Figure |
|---|---|---|---|
| 1 | Resource configuration comparison | `config_results` | Table 4 |
| 2 | Policy comparison (no-action / single / opportunistic) | `policy_results` | Table 5, Figure 3 |
| 3 | Sensitivity: degradation speed (α multiplier) | `sensitivity` | Figure 4a |
| 4 | Degradation trajectory (effective age & reliability) | `degradation_traj` | Figure 6 |
| 5 | Weather scenario summary | `weather_scenarios` | Table 3, Figure 7 |

---

## Generated Figures

| File | Content |
|---|---|
| `fig1_reliability.png` | Weibull survival curves + reliability–production coefficient mapping |
| `fig2_config_comparison.png` | Stacked cost breakdown by resource configuration |
| `fig3_policy_comparison.png` | Policy comparison bar chart + savings waterfall |
| `fig4_sensitivity.png` | Sensitivity to α multiplier and vessel accessibility |
| `fig5_maint_tech.png` | Maintenance level distribution + technician utilisation |
| `fig6_degradation.png` | Effective-age and reliability trajectories under three PM scenarios |
| `fig7_weather.png` | North Sea weather scenario characteristics |

---

## Model Parameters

Core parameters can be modified in `src/parameters.py`:

| Parameter | Default | Description |
|---|---|---|
| `N_TURBINES` | 10 | Number of turbines in the farm |
| `HORIZON` | 365 | Planning horizon (days) |
| `ELEC_PRICE` | 90.0 | Electricity price (USD/MWh) |
| `BETA` | [2.1, 1.8, 2.4] | Weibull shape per component |
| `ETA` | [1200, 900, 1500] | Weibull scale per component (days) |
| `PHI` | [0.70, 0.45, 0.05] | Age-reduction ratio per maintenance level |
| `R_MIN` / `R_FULL` | 0.30 / 0.85 | Production-coefficient thresholds |
| `MAINT_TRIGGER_R` | 0.62 | Reliability threshold to trigger maintenance |
| `WAVE_LIMIT` | 2.5 m | Vessel accessibility wave-height limit |

---

## Source Module Summary

### `src/parameters.py`
Centralised parameter store. All physical constants, cost figures, Weibull parameters, and weather scenarios are defined here. Import this module in all other scripts.

### `src/reliability.py`
- `weibull_survival(age, c)` — Weibull survival function $R_k(A) = \exp[-(A/\eta_k)^{\beta_k}]$
- `turbine_series_reliability(comp_rel)` — Series product $R_T = \prod_k R_k$
- `production_coefficient(R_T)` — Piecewise $C(R_T)$ mapping
- `effective_age_increment(age, c)` — Nonlinear degradation $\Delta A = 1 + \alpha(A^\gamma/\eta^\gamma)$
- `marginal_value_of_task(...)` — Forward-simulated production-gain estimate

### `src/patterns.py`
- `Pattern` dataclass — route + maintenance tasks + routing cost + score
- `generate_patterns(...)` — Opportunistic or single-visit pattern generation
- `pattern_score(...)` — Internal score for ranking candidate patterns
- `best_single_turbine_pattern(...)` — Used in the no-action gate

### `src/simulation.py`
- `simulate_scenario(...)` — Day-by-day rolling-horizon simulation for one weather scenario
- `evaluate_configuration(...)` — Probability-weighted evaluation across all scenarios

### `src/experiments.py`
- `run_config_comparison()` — Experiment 1
- `run_policy_comparison()` — Experiment 2
- `run_sensitivity_degradation()` — Experiment 3
- `run_degradation_trajectory()` — Experiment 4
- `run_all(output_dir)` — Orchestrates all experiments and writes JSON

---

## Key Results (Benchmark: 10-turbine farm, 5 North Sea scenarios)

| Metric | Value |
|---|---|
| Best configuration | Config A (1 Base / 2 Vessels / 6 Technicians) |
| Opportunistic total cost | ~$3.1M / year |
| Saving vs. single-visit policy | **~9%** |
| Saving vs. no-maintenance | **~52%** |
| Dominant cost driver | Production loss (>80% of total) |

---

## License

MIT License. See `LICENSE` for details.
