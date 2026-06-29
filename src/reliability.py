"""
reliability.py
==============
Weibull reliability, nonlinear degradation, and production coefficient
functions for the offshore wind O&M model.

Key equations (Section 3 of the paper):
  - Effective-age increment:    ΔA = 1 + α·(A^γ / η^γ)
  - Weibull survival:           R(A) = exp[−(A/η)^β]
  - Turbine series reliability: R_T = ∏_k R_k
  - Production coefficient:     C(R_T) = piecewise linear on [R_min, R_full]
  - Maintenance transition:     A⁺ = φ_ℓ · A⁻
"""

import math
import numpy as np
from parameters import BETA, ETA, ALPHA, GAMMA, PHI, R_MIN, R_FULL, N_COMPONENTS


# ── DEGRADATION ────────────────────────────────────────────────────────────

def effective_age_increment(age: float, c: int) -> float:
    """
    Compute the daily effective-age increment for component c.

    ΔA_k = 1 + α_k · (A_k)^γ_k / (η_k)^γ_k

    When α_k = 0 this reduces to 1 (linear aging).
    When α_k > 0 and γ_k > 1, aging accelerates nonlinearly.

    Parameters
    ----------
    age : float   Current effective age (days).
    c   : int     Component index (0=Gearbox, 1=Generator, 2=Rotor Blade).

    Returns
    -------
    float  Daily age increment.
    """
    return 1.0 + ALPHA[c] * (age ** GAMMA[c]) / (ETA[c] ** GAMMA[c])


def degrade_ages(ages: np.ndarray) -> np.ndarray:
    """
    Apply one day of degradation to a (n_turbines × n_components) age array.

    Parameters
    ----------
    ages : np.ndarray  Shape (n_turbines, n_components).

    Returns
    -------
    np.ndarray  Updated age array (same shape).
    """
    new_ages = ages.copy()
    n_turb, n_comp = ages.shape
    for i in range(n_turb):
        for c in range(n_comp):
            new_ages[i, c] += effective_age_increment(ages[i, c], c)
    return new_ages


def apply_maintenance(ages: np.ndarray, turbine: int, component: int, level: int) -> np.ndarray:
    """
    Apply imperfect maintenance to one component, returning updated age array.

    A_k^+ = φ_ℓ · A_k^−

    Parameters
    ----------
    ages      : np.ndarray  Shape (n_turbines, n_components).
    turbine   : int         Turbine index.
    component : int         Component index.
    level     : int         Maintenance level index (0=L1, 1=L2, 2=L3).

    Returns
    -------
    np.ndarray  Updated age array.
    """
    new_ages = ages.copy()
    new_ages[turbine, component] *= PHI[level]
    return new_ages


# ── RELIABILITY ────────────────────────────────────────────────────────────

def weibull_survival(age: float, c: int) -> float:
    """
    Weibull survival (cumulative) reliability for component c at given age.

    R_k(A) = exp[−(A / η_k)^β_k]

    Parameters
    ----------
    age : float  Effective age (days).
    c   : int    Component index.

    Returns
    -------
    float  Survival reliability ∈ [0, 1].
    """
    if age <= 0:
        return 1.0
    return math.exp(-((age / ETA[c]) ** BETA[c]))


def component_reliabilities(ages: np.ndarray) -> np.ndarray:
    """
    Compute Weibull survival reliability for all turbines and components.

    Parameters
    ----------
    ages : np.ndarray  Shape (n_turbines, n_components).

    Returns
    -------
    np.ndarray  Shape (n_turbines, n_components), values ∈ [0, 1].
    """
    n_turb, n_comp = ages.shape
    R = np.zeros((n_turb, n_comp))
    for i in range(n_turb):
        for c in range(n_comp):
            R[i, c] = weibull_survival(ages[i, c], c)
    return R


def turbine_series_reliability(comp_rel: np.ndarray) -> np.ndarray:
    """
    Turbine reliability as a series system (product of component reliabilities).

    R_T_i = ∏_{k=1}^{K} R_k_i

    Parameters
    ----------
    comp_rel : np.ndarray  Shape (n_turbines, n_components).

    Returns
    -------
    np.ndarray  Shape (n_turbines,), turbine-level reliability.
    """
    return np.prod(comp_rel, axis=1)


def production_coefficient(R_T: float) -> float:
    """
    Piecewise-linear reliability–production coefficient.

    C(R_T) = 1                           if R_T ≥ R_full
           = (R_T − R_min)/(R_full − R_min)  if R_min < R_T < R_full
           = 0                           if R_T ≤ R_min

    Parameters
    ----------
    R_T : float  Turbine-level survival reliability.

    Returns
    -------
    float  Production coefficient ∈ [0, 1].
    """
    if R_T >= R_FULL:
        return 1.0
    if R_T <= R_MIN:
        return 0.0
    return (R_T - R_MIN) / (R_FULL - R_MIN)


def production_coefficients(turb_rel: np.ndarray) -> np.ndarray:
    """
    Vectorized production coefficient for an array of turbine reliabilities.

    Parameters
    ----------
    turb_rel : np.ndarray  Shape (n_turbines,).

    Returns
    -------
    np.ndarray  Shape (n_turbines,), values ∈ [0, 1].
    """
    return np.array([production_coefficient(r) for r in turb_rel])


# ── PRODUCTION LOSS ─────────────────────────────────────────────────────────

def daily_production_loss(
    comp_rel: np.ndarray,
    daily_potential_mwh: float,
    elec_price: float,
    scenario_prob: float,
) -> float:
    """
    Expected daily production-loss cost across all turbines.

    PL = Σ_i  w_s · π · P̂_t · (1 − C(R_T_i))

    Parameters
    ----------
    comp_rel            : np.ndarray  Shape (n_turbines, n_components).
    daily_potential_mwh : float       Daily potential production per turbine (MWh).
    elec_price          : float       Electricity price (USD/MWh).
    scenario_prob       : float       Scenario probability weight.

    Returns
    -------
    float  Expected production-loss cost (USD).
    """
    R_T = turbine_series_reliability(comp_rel)
    C   = production_coefficients(R_T)
    n_turb = comp_rel.shape[0]
    loss = 0.0
    for i in range(n_turb):
        # If R_T ≤ R_min, production is zero → full potential is lost
        loss += scenario_prob * elec_price * daily_potential_mwh * (1.0 - C[i])
    return loss


# ── MARGINAL VALUE OF A MAINTENANCE TASK ───────────────────────────────────

def marginal_value_of_task(
    ages: np.ndarray,
    turbine: int,
    component: int,
    level: int,
    daily_potential_mwh: float,
    elec_price: float,
    forecast_days: int = 30,
) -> float:
    """
    Approximate marginal value of performing maintenance task (i, k, ℓ).

    MV ≈ Σ_{τ=1}^{W} π · P̂ · ΔC(R_T_i, τ) − c^maint_ℓ

    Degradation is simulated forward W days with and without the task to
    estimate the expected production-coefficient improvement.

    Parameters
    ----------
    ages                : np.ndarray  Current age array (n_turbines, n_components).
    turbine             : int         Target turbine.
    component           : int         Target component.
    level               : int         Maintenance level (0=L1, 1=L2, 2=L3).
    daily_potential_mwh : float       Daily potential production (MWh/turbine).
    elec_price          : float       Electricity price (USD/MWh).
    forecast_days       : int         Rolling-horizon forecast window W.

    Returns
    -------
    float  Approximate marginal value (USD). Positive → maintenance is beneficial.
    """
    from parameters import MAINT_COST_USD

    ages_no   = ages.copy()
    ages_with = apply_maintenance(ages.copy(), turbine, component, level)

    gain = 0.0
    for _ in range(forecast_days):
        C_no   = production_coefficient(
            math.prod(weibull_survival(ages_no[turbine, c], c)   for c in range(N_COMPONENTS))
        )
        C_with = production_coefficient(
            math.prod(weibull_survival(ages_with[turbine, c], c) for c in range(N_COMPONENTS))
        )
        gain += elec_price * daily_potential_mwh * (C_with - C_no)
        ages_no   = degrade_ages(ages_no)
        ages_with = degrade_ages(ages_with)

    mv = gain - MAINT_COST_USD[level]
    return mv


# ── UTILITY: power curve ────────────────────────────────────────────────────

def power_curve(wind_speed: float, cut_in: float = 3.0,
                rated: float = 12.0, cut_out: float = 25.0) -> float:
    """
    Simplified cubic power curve (normalized to rated power).

    Returns fraction of rated power ∈ [0, 1].
    """
    if wind_speed < cut_in or wind_speed >= cut_out:
        return 0.0
    if wind_speed >= rated:
        return 1.0
    return (wind_speed - cut_in) / (rated - cut_in)


def daily_potential_mwh(wind_speed: float, rated_mw: float = 3.0) -> float:
    """Daily potential energy output per turbine at given mean wind speed (MWh)."""
    return 24.0 * rated_mw * power_curve(wind_speed)
