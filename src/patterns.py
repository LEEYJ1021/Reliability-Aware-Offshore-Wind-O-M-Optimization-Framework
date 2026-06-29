"""
patterns.py
===========
Route–maintenance pattern generation for the offshore wind O&M model.

A pattern p = (ρ, ℳ) consists of:
  ρ  – vessel route: sequence of turbines starting and ending at base
  ℳ  – set of component-level maintenance tasks (turbine, component, level)

Routing cost is charged once to the entire pattern. Incremental tasks on
additional turbines within the same route incur only marginal maintenance
and downtime costs.

Sections referenced: 3.5, 3.21.2, 3.21.3 of the paper.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from parameters import (
    DIST_BASE_TO_TURB, DIST_INTER_TURB, VESSEL_SPEED_KPH,
    MAINT_COST_USD, MAINT_TIME_H_STD, MAX_TURBINES_ROUTE,
    MAINT_TRIGGER_R, OPP_BUNDLE_R, ACC_GATE,
    ELEC_PRICE, P_RATED_MW, HORIZON, N_COMPONENTS,
)
from reliability import (
    turbine_series_reliability, production_coefficient,
    weibull_survival, degrade_ages, apply_maintenance, daily_potential_mwh,
)


# ── DATA CLASSES ───────────────────────────────────────────────────────────

@dataclass
class MaintenanceTask:
    """Single component-level maintenance action within a pattern."""
    turbine:   int    # turbine index i
    component: int    # component index k
    level:     int    # maintenance level ℓ (0=L1, 1=L2, 2=L3)
    maint_cost: float = field(init=False)
    task_time_h: float = field(init=False)

    def __post_init__(self):
        self.maint_cost  = MAINT_COST_USD[self.level]
        self.task_time_h = MAINT_TIME_H_STD[self.level]


@dataclass
class Pattern:
    """
    A route–maintenance pattern.

    route         : ordered list of turbine indices visited
    tasks         : list of MaintenanceTask objects
    routing_cost  : total vessel routing cost (charged once to the pattern)
    maint_cost    : sum of direct maintenance costs across all tasks
    score         : internal pattern score (used for ranking, not objective)
    """
    route:        List[int]
    tasks:        List[MaintenanceTask]
    routing_cost: float = 0.0
    maint_cost:   float = 0.0
    score:        float = 0.0

    def turbines_visited(self) -> List[int]:
        return list(dict.fromkeys(self.route))   # preserve order, no duplicates

    def tasks_for_turbine(self, turbine: int) -> List[MaintenanceTask]:
        return [t for t in self.tasks if t.turbine == turbine]

    def total_task_time_h(self) -> float:
        return sum(t.task_time_h for t in self.tasks)

    def total_cost(self) -> float:
        return self.routing_cost + self.maint_cost

    def is_null(self) -> bool:
        return len(self.tasks) == 0


# ── ROUTING COST CALCULATION ───────────────────────────────────────────────

def compute_routing_cost(n_turbines_in_route: int) -> float:
    """
    Approximate routing cost for a route visiting n turbines.

    Cost = (base_distance + inter_turbine_distance × (n−1)) × vessel_cost_per_km

    Parameters
    ----------
    n_turbines_in_route : int  Number of turbines visited.

    Returns
    -------
    float  Routing cost (USD).
    """
    if n_turbines_in_route == 0:
        return 0.0
    dist = DIST_BASE_TO_TURB + DIST_INTER_TURB * max(0, n_turbines_in_route - 1)
    return dist * VESSEL_SPEED_KPH   # USD (vessel_speed_kph used as cost-per-km proxy)


# ── PATTERN SCORE ──────────────────────────────────────────────────────────

def pattern_score(
    pattern: Pattern,
    ages: np.ndarray,
    daily_pot_mwh: float,
    scenario_prob: float,
    forecast_days: int = 14,
) -> float:
    """
    Internal pattern score: approximate net benefit of executing the pattern.

    Score = Σ_{tasks} MV(task) − routing_cost_share

    The routing cost is not deducted per task but is shared across all tasks
    (a single vessel dispatch covers all tasks in the pattern).

    Parameters
    ----------
    pattern       : Pattern
    ages          : np.ndarray  Current age array (n_turbines, n_components).
    daily_pot_mwh : float       Daily potential production per turbine (MWh).
    scenario_prob : float       Scenario probability weight.
    forecast_days : int         Horizon for production-gain estimation.

    Returns
    -------
    float  Pattern score (higher is better).
    """
    if not pattern.tasks:
        return 0.0

    ages_no   = ages.copy()
    ages_with = ages.copy()
    for task in pattern.tasks:
        ages_with = apply_maintenance(ages_with, task.turbine, task.component, task.level)

    total_gain = 0.0
    for d in range(forecast_days):
        for task in pattern.tasks:
            i = task.turbine
            R_no   = math.prod(weibull_survival(ages_no[i, c], c)   for c in range(N_COMPONENTS))
            R_with = math.prod(weibull_survival(ages_with[i, c], c) for c in range(N_COMPONENTS))
            C_no   = production_coefficient(R_no)
            C_with = production_coefficient(R_with)
            total_gain += scenario_prob * ELEC_PRICE * daily_pot_mwh * (C_with - C_no)
        ages_no   = degrade_ages(ages_no)
        ages_with = degrade_ages(ages_with)

    pm_downtime_cost = sum(
        scenario_prob * ELEC_PRICE * daily_pot_mwh * (task.task_time_h / 24.0)
        for task in pattern.tasks
    )

    score = total_gain - pattern.maint_cost - pm_downtime_cost - pattern.routing_cost
    return score


# ── OPPORTUNISTIC PATTERN GENERATION ──────────────────────────────────────

def generate_patterns(
    ages: np.ndarray,
    comp_rel: np.ndarray,
    turb_rel: np.ndarray,
    daily_pot_mwh: float,
    scenario_prob: float,
    n_vessels: int,
    n_tech: int,
    accessibility: float,
    policy: str = "opportunistic",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Pattern], List[int]]:
    """
    Generate candidate route–maintenance patterns for the current decision epoch.

    Implements Section 3.21.2–3.21.3 of the paper:
      1. Identify seed turbines (R_T < MAINT_TRIGGER_R) ordered by reliability.
      2. For each seed, select worst component and appropriate maintenance level.
      3. In opportunistic mode, add nearby turbines (R_T < OPP_BUNDLE_R).
      4. Compute routing cost, maintenance cost, and pattern score.

    Parameters
    ----------
    ages          : np.ndarray  Shape (n_turbines, n_components).
    comp_rel      : np.ndarray  Shape (n_turbines, n_components).
    turb_rel      : np.ndarray  Shape (n_turbines,).
    daily_pot_mwh : float
    scenario_prob : float
    n_vessels     : int
    n_tech        : int
    accessibility : float       Scenario vessel accessibility fraction.
    policy        : str         "opportunistic" | "single" | "no_action"
    rng           : np.random.Generator | None

    Returns
    -------
    patterns      : List[Pattern]  Candidate patterns (may be empty).
    visited_today : List[int]      Turbine indices in the executed pattern.
    """
    if policy == "no_action":
        return [], []

    if accessibility < ACC_GATE:
        return [], []

    max_turbines_today = min(
        ages.shape[0],
        n_vessels * 2,
        n_tech // 2,
    )

    # --- Seed turbines (sorted worst-first by turbine reliability) ----------
    order = np.argsort(turb_rel)
    seeds = [int(i) for i in order if turb_rel[i] < MAINT_TRIGGER_R]
    seeds = seeds[:max_turbines_today]

    if not seeds:
        return [], []

    # --- Build bundle -------------------------------------------------------
    if policy == "opportunistic":
        visited_set = set(seeds)
        extra = [
            int(i) for i in range(ages.shape[0])
            if i not in visited_set
            and turb_rel[i] < OPP_BUNDLE_R
            and len(visited_set) < max_turbines_today
        ]
        bundle = seeds + extra
    else:
        bundle = seeds   # single-visit: only seed turbines

    # Enforce maximum turbines per route
    bundle = bundle[:MAX_TURBINES_ROUTE]

    # --- Construct tasks for each turbine in the bundle --------------------
    tasks = []
    for turb_idx in bundle:
        worst_comp = int(np.argmin(comp_rel[turb_idx]))
        age_wc = ages[turb_idx, worst_comp]
        # Choose maintenance level based on component age
        if age_wc > 800:
            level = 2   # L3 (major)
        elif age_wc > 600:
            level = 1   # L2 (moderate)
        else:
            level = 0   # L1 (minor)
        tasks.append(MaintenanceTask(turbine=turb_idx, component=worst_comp, level=level))

    # --- Routing cost (shared across all tasks in the bundle) --------------
    routing_cost = compute_routing_cost(len(bundle))
    maint_cost   = sum(t.maint_cost for t in tasks)

    pattern = Pattern(
        route=bundle,
        tasks=tasks,
        routing_cost=routing_cost,
        maint_cost=maint_cost,
    )
    pattern.score = pattern_score(pattern, ages, daily_pot_mwh, scenario_prob)

    return [pattern], bundle


# ── NULL PATTERN (no action) ───────────────────────────────────────────────

NULL_PATTERN = Pattern(route=[], tasks=[], routing_cost=0.0, maint_cost=0.0, score=0.0)


# ── SINGLE-TURBINE BEST PATTERN ────────────────────────────────────────────

def best_single_turbine_pattern(
    ages: np.ndarray,
    comp_rel: np.ndarray,
    turb_rel: np.ndarray,
    daily_pot_mwh: float,
    scenario_prob: float,
) -> Optional[Pattern]:
    """
    Find the best single-turbine, single-component pattern by pattern score.
    Used in the no-action gate comparison.
    """
    best: Optional[Pattern] = None
    best_score = -1e18
    n_turb = ages.shape[0]
    for i in range(n_turb):
        for c in range(N_COMPONENTS):
            for lvl in range(3):
                tasks = [MaintenanceTask(turbine=i, component=c, level=lvl)]
                rc = compute_routing_cost(1)
                mc = MAINT_COST_USD[lvl]
                p  = Pattern(route=[i], tasks=tasks, routing_cost=rc, maint_cost=mc)
                p.score = pattern_score(p, ages, daily_pot_mwh, scenario_prob)
                if p.score > best_score:
                    best_score = p.score
                    best = p
    return best
