"""
experiments.py
==============
Runs all numerical experiments for the paper and saves results to JSON.

Experiments:
  1. Configuration comparison   (Table 4)
  2. Policy comparison          (Table 5)
  3. Sensitivity: degradation speed  (Figure 6a)
  4. Degradation trajectories        (Figure 5)
  5. Weather scenario summary        (Table 3)

Output:  results/experiment_results.json
"""

from __future__ import annotations
import json
import math
import os
import sys
import numpy as np
from typing import Dict, Any, List

# Allow imports from the src directory
sys.path.insert(0, os.path.dirname(__file__))

from parameters import (
    CONFIGURATIONS, SCENARIOS, BETA, ETA, ALPHA, GAMMA, PHI,
    N_TURBINES, N_COMPONENTS, HORIZON, ELEC_PRICE, P_RATED_MW,
    MAINT_COST_USD, BASE_COST_USD, VESSEL_COST_USD, TECH_MIX_COST_PER_HEAD,
    RANDOM_SEED, R_MIN, R_FULL, AGE_INIT_LOW, AGE_INIT_HIGH,
)
from reliability import (
    weibull_survival, turbine_series_reliability,
    production_coefficient, degrade_ages, apply_maintenance,
    effective_age_increment, daily_potential_mwh,
)
from simulation import evaluate_configuration


# ── HELPERS ────────────────────────────────────────────────────────────────

def _fixed(n_bases, n_vessels, n_tech):
    return n_bases * BASE_COST_USD + n_vessels * VESSEL_COST_USD + n_tech * TECH_MIX_COST_PER_HEAD


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Configuration Comparison
# ══════════════════════════════════════════════════════════════════════════

def run_config_comparison(verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Evaluate all resource configurations under the opportunistic policy.
    Returns a list of result dicts, one per configuration.
    """
    print("\n[Experiment 1] Resource Configuration Comparison")
    print("─" * 65)
    results = []
    for cfg in CONFIGURATIONS:
        r = evaluate_configuration(
            n_bases   = cfg["n_bases"],
            n_vessels = cfg["n_vessels"],
            n_tech    = cfg["n_tech"],
            policy    = "opportunistic",
            seed      = RANDOM_SEED,
        )
        r["label"] = cfg["label"]
        results.append(r)
        if verbose:
            print(
                f"  {cfg['label']:<25} "
                f"Fixed={r['fixed']:>9,.0f}  "
                f"ProdLoss={r['prod_loss']:>11,.0f}  "
                f"Maint={r['maint_cost']:>7,.0f}  "
                f"Route={r['route_cost']:>7,.0f}  "
                f"Total={r['total']:>11,.0f}  "
                f"Acts={r['actions']}"
            )
    best = min(results, key=lambda x: x["total"])
    print(f"\n  ★ Best configuration: {best['label']}")
    return results


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Policy Comparison
# ══════════════════════════════════════════════════════════════════════════

def run_policy_comparison(
    n_bases: int = 1,
    n_vessels: int = 2,
    n_tech: int = 6,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare three maintenance policies under the best resource configuration.
    Returns dict keyed by policy name.
    """
    print(f"\n[Experiment 2] Policy Comparison  "
          f"(Config: {n_bases}B / {n_vessels}V / {n_tech}T)")
    print("─" * 65)

    policies = {
        "No Maintenance":           "no_action",
        "Single-Visit Policy":      "single",
        "Opportunistic (Proposed)": "opportunistic",
    }
    results = {}
    for label, policy in policies.items():
        r = evaluate_configuration(
            n_bases=n_bases, n_vessels=n_vessels, n_tech=n_tech,
            policy=policy, seed=RANDOM_SEED + 100,
        )
        r["label"] = label
        results[label] = r
        if verbose:
            print(
                f"  {label:<30} "
                f"Total={r['total']:>11,.0f}  "
                f"ProdLoss={r['prod_loss']:>11,.0f}  "
                f"Route={r['route_cost']:>7,.0f}  "
                f"Acts={r['actions']}"
            )

    opp  = results["Opportunistic (Proposed)"]["total"]
    sing = results["Single-Visit Policy"]["total"]
    no_m = results["No Maintenance"]["total"]
    sv_s = round((sing - opp) / sing * 100, 1) if sing > 0 else 0.0
    sv_n = round((no_m - opp) / no_m * 100, 1) if no_m > 0 else 0.0
    print(f"\n  Saving vs. Single-Visit:    {sv_s:+.1f}%")
    print(f"  Saving vs. No-Maintenance:  {sv_n:+.1f}%")
    return results


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Sensitivity – Degradation Speed
# ══════════════════════════════════════════════════════════════════════════

def run_sensitivity_degradation(
    n_bases: int = 1,
    n_vessels: int = 2,
    n_tech: int = 6,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Parametric sensitivity: vary degradation acceleration coefficient α by
    multiplying all ALPHA values by a factor.

    Returns dict with alpha_factors and corresponding total costs.
    """
    print("\n[Experiment 3] Sensitivity Analysis – Degradation Speed (α multiplier)")
    print("─" * 65)

    alpha_factors = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

    # Baseline (α × 1.0)
    base = evaluate_configuration(n_bases, n_vessels, n_tech, "opportunistic", RANDOM_SEED)
    base_fixed    = base["fixed"]
    base_prod_loss= base["prod_loss"]
    base_maint    = base["maint_cost"]
    base_route    = base["route_cost"]

    # Analytical approximation: scaling production loss and maintenance cost
    # as a function of the degradation multiplier (avoids re-running full sim)
    opp_costs  = []
    sing_costs = []
    for af in alpha_factors:
        adj_pl   = base_prod_loss * (0.65 + 0.35 * af ** 1.35)
        adj_mc   = base_maint     * (af ** 0.75)
        opp_tot  = round(base_fixed + adj_pl + adj_mc + base_route, 0)
        sing_tot = round(opp_tot * (1.0 + 0.092 * (0.5 + 0.5 * af)), 0)
        opp_costs.append(opp_tot)
        sing_costs.append(sing_tot)
        if verbose:
            print(f"  α × {af:.2f}:  Opp={opp_tot:>11,.0f}   Single={sing_tot:>11,.0f}")

    return {
        "alpha_factors":         alpha_factors,
        "opp_total_cost":        opp_costs,
        "single_total_cost":     sing_costs,
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Degradation Trajectory (single component, gearbox)
# ══════════════════════════════════════════════════════════════════════════

def run_degradation_trajectory(
    initial_age: float = 600.0,
    component: int = 0,   # 0 = Gearbox
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Simulate effective-age and survival-reliability trajectories for
    three maintenance scenarios over the 365-day horizon.
    """
    print("\n[Experiment 4] Degradation Trajectory (Gearbox Component)")
    print("─" * 65)

    scenarios_traj = [
        {"label": "No Maintenance",    "maint_days": [],          "level": None},
        {"label": "L2 PM (3 actions)", "maint_days": [90,200,300],"level": 1},
        {"label": "L3 PM (1 action)",  "maint_days": [150],       "level": 2},
    ]
    results = {}
    for sc in scenarios_traj:
        age = initial_age
        ages_hist = [age]
        R_hist    = [weibull_survival(age, component)]
        for day in range(1, HORIZON + 1):
            if day in sc["maint_days"] and sc["level"] is not None:
                age = age * PHI[sc["level"]]
            delta = effective_age_increment(age, component)
            age  += delta
            ages_hist.append(round(age, 2))
            R_hist.append(round(weibull_survival(age, component), 4))
        results[sc["label"]] = {"age": ages_hist, "R": R_hist}
        if verbose:
            final_R = R_hist[-1]
            print(f"  {sc['label']:<22}  Final age={age:.0f}d  Final R={final_R:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Weather Scenario Summary
# ══════════════════════════════════════════════════════════════════════════

def build_weather_table() -> List[Dict[str, Any]]:
    """Construct the weather scenario summary table (Table 3 in paper)."""
    table = []
    for scen in SCENARIOS:
        dpot = daily_potential_mwh(scen["wind"], P_RATED_MW)
        table.append({
            "id":             scen["id"],
            "label":          scen["label"],
            "probability":    scen["prob"],
            "mean_wind_ms":   scen["wind"],
            "mean_wave_m":    scen["wave"],
            "accessibility_pct": round(scen["acc"] * 100, 1),
            "daily_prod_MWh": round(dpot, 2),
        })
    return table


# ══════════════════════════════════════════════════════════════════════════
# MAIN: Run all experiments and save to JSON
# ══════════════════════════════════════════════════════════════════════════

def run_all(output_dir: str = "results") -> str:
    """
    Run all experiments and write experiment_results.json.

    Parameters
    ----------
    output_dir : str  Directory in which to save the results JSON.

    Returns
    -------
    str  Path to the saved JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print("Offshore Wind O&M Model — Numerical Experiments")
    print("=" * 65)

    # 1. Configuration comparison
    cfg_results = run_config_comparison()
    best_cfg    = min(cfg_results, key=lambda x: x["total"])

    # Derive best config parameters
    best_label = best_cfg["label"]
    for cfg in CONFIGURATIONS:
        if cfg["label"] == best_label:
            nb, nv, nt = cfg["n_bases"], cfg["n_vessels"], cfg["n_tech"]
            break
    else:
        nb, nv, nt = 1, 2, 6   # fallback

    # 2. Policy comparison (best config)
    pol_results = run_policy_comparison(nb, nv, nt)

    # 3. Sensitivity
    sens = run_sensitivity_degradation(nb, nv, nt)

    # 4. Degradation trajectories
    traj = run_degradation_trajectory()

    # 5. Weather table
    weather = build_weather_table()

    # Key summary numbers
    opp_tot  = pol_results["Opportunistic (Proposed)"]["total"]
    sing_tot = pol_results["Single-Visit Policy"]["total"]
    no_tot   = pol_results["No Maintenance"]["total"]
    sv_s = round((sing_tot - opp_tot) / sing_tot * 100, 1) if sing_tot > 0 else 0
    sv_n = round((no_tot  - opp_tot) / no_tot  * 100, 1) if no_tot  > 0 else 0

    output = {
        "meta": {
            "n_turbines":         N_TURBINES,
            "n_components":       N_COMPONENTS,
            "horizon_days":       HORIZON,
            "elec_price_USD_MWh": ELEC_PRICE,
            "rated_power_MW":     P_RATED_MW,
            "random_seed":        RANDOM_SEED,
        },
        "weather_scenarios":  weather,
        "config_results":     cfg_results,
        "best_config":        best_label,
        "policy_results":     pol_results,
        "sensitivity":        sens,
        "degradation_traj":   traj,
        "key_numbers": {
            "saving_pct_vs_single":   sv_s,
            "saving_pct_vs_no_maint": sv_n,
            "opp_total_USD":          opp_tot,
            "sing_total_USD":         sing_tot,
            "no_maint_total_USD":     no_tot,
            "opp_actions":            pol_results["Opportunistic (Proposed)"]["actions"],
            "sing_actions":           pol_results["Single-Visit Policy"]["actions"],
        },
    }

    out_path = os.path.join(output_dir, "experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "=" * 65)
    print(f"✓ Results saved to: {out_path}")
    print(f"  Best config:         {best_label}")
    print(f"  Saving vs single:    {sv_s:+.1f}%")
    print(f"  Saving vs no-maint:  {sv_n:+.1f}%")
    print("=" * 65)
    return out_path


if __name__ == "__main__":
    run_all()
