"""
simulation.py
=============
Rolling-horizon simulation engine for the offshore wind O&M model.

Implements the two-stage stochastic evaluation:
  - First stage:  fixed resource configuration (n_bases, n_vessels, n_tech)
  - Second stage: rolling-horizon LNS with no-action gate, evaluated across
                  all weather scenarios weighted by probability

Section references: 3.19–3.21 of the paper.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any

from parameters import (
    N_TURBINES, N_COMPONENTS, HORIZON, ELEC_PRICE, P_RATED_MW,
    SCENARIOS, BASE_COST_USD, VESSEL_COST_USD, TECH_MIX_COST_PER_HEAD,
    RANDOM_SEED, MAINT_COST_USD, AGE_INIT_LOW, AGE_INIT_HIGH,
)
from reliability import (
    component_reliabilities, turbine_series_reliability,
    production_coefficients, degrade_ages, apply_maintenance,
    daily_potential_mwh,
)
from patterns import (
    generate_patterns, best_single_turbine_pattern,
    NULL_PATTERN, compute_routing_cost,
)


# ── COST OF A PLAN ─────────────────────────────────────────────────────────

def plan_daily_cost(
    ages: np.ndarray,
    comp_rel: np.ndarray,
    bundle: list,
    tasks: list,
    routing_cost: float,
    maint_cost: float,
    daily_pot_mwh: float,
    scenario_prob: float,
) -> float:
    """
    Compute the actual daily cost of an executed plan.

    Cost = routing_cost + maint_cost + production_loss

    Used in the no-action gate (Section 3.21.5).
    """
    turb_rel = turbine_series_reliability(comp_rel)
    pcs = production_coefficients(turb_rel)
    prod_loss = sum(
        scenario_prob * ELEC_PRICE * daily_pot_mwh * (1.0 - pcs[i])
        for i in range(N_TURBINES)
    )
    return routing_cost * scenario_prob + maint_cost * scenario_prob + prod_loss


# ── SINGLE-SCENARIO SIMULATION ─────────────────────────────────────────────

def simulate_scenario(
    n_vessels: int,
    n_tech: int,
    s_prob: float,
    s_acc: float,
    s_wind: float,
    policy: str,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Simulate one weather scenario over the full planning horizon.

    Parameters
    ----------
    n_vessels : int     Number of vessels.
    n_tech    : int     Number of technicians.
    s_prob    : float   Scenario probability weight.
    s_acc     : float   Vessel accessibility fraction (from wave height).
    s_wind    : float   Mean wind speed (m/s).
    policy    : str     "no_action" | "single" | "opportunistic"
    rng       : np.random.Generator

    Returns
    -------
    dict with keys: prod_loss, maint_cost, route_cost, actions
    """
    dpot = daily_potential_mwh(s_wind, P_RATED_MW)

    # Initialize ages: uniform draw across turbines
    ages = rng.uniform(AGE_INIT_LOW, AGE_INIT_HIGH, (N_TURBINES, N_COMPONENTS)).astype(float)

    tot_pl = 0.0
    tot_mc = 0.0
    tot_rc = 0.0
    n_actions = 0

    for day in range(HORIZON):

        # ── Reliability & production loss ──────────────────────────────────
        comp_rel = component_reliabilities(ages)
        turb_rel = turbine_series_reliability(comp_rel)
        pcs      = production_coefficients(turb_rel)

        for i in range(N_TURBINES):
            tot_pl += s_prob * ELEC_PRICE * dpot * (1.0 - pcs[i])

        # ── Maintenance decision ───────────────────────────────────────────
        if policy != "no_action":
            patterns, bundle = generate_patterns(
                ages=ages,
                comp_rel=comp_rel,
                turb_rel=turb_rel,
                daily_pot_mwh=dpot,
                scenario_prob=s_prob,
                n_vessels=n_vessels,
                n_tech=n_tech,
                accessibility=s_acc,
                policy=policy,
                rng=rng,
            )

            if patterns and bundle:
                pattern = patterns[0]   # take the single generated pattern

                # ── No-action gate (Section 3.21.5) ────────────────────────
                # Compare proposed plan against no-action and single-turbine plan.
                # In rolling-horizon we approximate using pattern.score vs 0.
                execute = pattern.score > 0

                if execute:
                    # Apply all maintenance tasks in the pattern
                    for task in pattern.tasks:
                        ages = apply_maintenance(
                            ages, task.turbine, task.component, task.level
                        )
                        tot_mc += task.maint_cost * s_prob
                        n_actions += 1

                    # Routing cost is charged once per dispatch
                    tot_rc += pattern.routing_cost * s_prob

        # ── Degrade all components ─────────────────────────────────────────
        ages = degrade_ages(ages)

    return {
        "prod_loss":  round(tot_pl, 2),
        "maint_cost": round(tot_mc, 2),
        "route_cost": round(tot_rc, 2),
        "actions":    n_actions,
    }


# ── FULL CONFIGURATION EVALUATION ──────────────────────────────────────────

def evaluate_configuration(
    n_bases: int,
    n_vessels: int,
    n_tech: int,
    policy: str = "opportunistic",
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Evaluate a resource configuration under all weather scenarios.

    Two-stage stochastic evaluation:
      Total cost = Fixed cost + Σ_s w_s · Operational cost(s)

    Parameters
    ----------
    n_bases   : int   Number of O&M bases opened.
    n_vessels : int   Number of vessels deployed.
    n_tech    : int   Number of technicians.
    policy    : str   Maintenance policy for the second stage.
    seed      : int   Random seed for reproducibility.

    Returns
    -------
    dict with keys: fixed, prod_loss, maint_cost, route_cost, total, actions
    """
    fixed_cost = (
        n_bases   * BASE_COST_USD
        + n_vessels * VESSEL_COST_USD
        + n_tech    * TECH_MIX_COST_PER_HEAD
    )

    total_pl = 0.0
    total_mc = 0.0
    total_rc = 0.0
    total_actions = 0

    for s_idx, scen in enumerate(SCENARIOS):
        rng = np.random.default_rng(seed + s_idx * 17)
        res = simulate_scenario(
            n_vessels=n_vessels,
            n_tech=n_tech,
            s_prob=scen["prob"],
            s_acc=scen["acc"],
            s_wind=scen["wind"],
            policy=policy,
            rng=rng,
        )
        total_pl      += res["prod_loss"]
        total_mc      += res["maint_cost"]
        total_rc      += res["route_cost"]
        total_actions += res["actions"]

    total_cost = fixed_cost + total_pl + total_mc + total_rc

    return {
        "fixed":      round(fixed_cost, 0),
        "prod_loss":  round(total_pl,   0),
        "maint_cost": round(total_mc,   0),
        "route_cost": round(total_rc,   0),
        "total":      round(total_cost, 0),
        "actions":    total_actions,
    }
