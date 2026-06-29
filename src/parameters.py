"""
parameters.py
=============
Shared parameters for the offshore wind O&M model.

Reference:
  "Reliability-Based Maintenance Resource Design and Opportunistic Routing
   Optimization for Offshore Wind Farms under Weather Uncertainty"

All monetary values in USD. Time in days. Distance in km.
"""

import numpy as np

# ── FARM ──────────────────────────────────────────────────────────────────
N_TURBINES   = 10       # |ℐ|
N_COMPONENTS = 3        # components per turbine: gearbox, generator, rotor blade
HORIZON      = 365      # planning horizon (days)
P_RATED_MW   = 3.0      # rated power per turbine (MW)
ELEC_PRICE   = 90.0     # electricity selling price (USD/MWh)

# ── WEIBULL COMPONENT PARAMETERS ──────────────────────────────────────────
# Index order: 0=Gearbox, 1=Generator, 2=Rotor Blade
COMP_NAMES = ["Gearbox", "Generator", "Rotor Blade"]
BETA       = np.array([2.1,   1.8,   2.4])    # Weibull shape  β_k
ETA        = np.array([1200., 900.,  1500.])   # Weibull scale  η_k (days)
ALPHA      = np.array([0.020, 0.030, 0.015])   # degradation acceleration α_k
GAMMA      = np.array([1.50,  1.60,  1.40])    # degradation exponent     γ_k

# ── INITIAL EFFECTIVE AGES (uniform range, mid-to-late life) ──────────────
AGE_INIT_LOW  = np.array([600., 500., 700.])   # days
AGE_INIT_HIGH = np.array([900., 750., 1000.])  # days

# ── MAINTENANCE LEVELS ────────────────────────────────────────────────────
# PHI[ℓ] = effective-age remaining ratio after maintenance level ℓ (0=L1,1=L2,2=L3)
MAINT_LEVELS     = ["L1", "L2", "L3"]
PHI              = np.array([0.70, 0.45, 0.05])   # φ_{ℓk} (same for all k)
MAINT_COST_USD   = np.array([350., 750., 1500.])  # direct maintenance cost per action
MAINT_TIME_H_STD = np.array([3.5,  6.0,  10.0])  # processing time (Standard tech, hours)

# ── RELIABILITY–PRODUCTION COEFFICIENT THRESHOLDS ────────────────────────
R_MIN  = 0.30   # below this → production coefficient = 0
R_FULL = 0.85   # above this → production coefficient = 1

# ── WEATHER SCENARIOS (K-medoids clusters from North Sea historical data) ─
# Each scenario: (probability, mean wind m/s, mean wave m, accessibility ratio)
SCENARIOS = [
    {"id": "S1", "label": "Mild",           "prob": 0.25, "wind": 7.5,  "wave": 1.2, "acc": 0.975},
    {"id": "S2", "label": "Moderate",       "prob": 0.30, "wind": 8.2,  "wave": 1.8, "acc": 0.850},
    {"id": "S3", "label": "Moderate-High",  "prob": 0.20, "wind": 9.1,  "wave": 2.5, "acc": 0.625},
    {"id": "S4", "label": "High",           "prob": 0.15, "wind": 10.5, "wave": 3.2, "acc": 0.375},
    {"id": "S5", "label": "Severe",         "prob": 0.10, "wind": 12.3, "wave": 4.0, "acc": 0.125},
]
WAVE_LIMIT = 2.5   # vessel accessibility wave-height limit (m)

# ── POWER CURVE ────────────────────────────────────────────────────────────
WS_CUT_IN  = 3.0    # m/s
WS_RATED   = 12.0   # m/s
WS_CUT_OUT = 25.0   # m/s

# ── RESOURCE COST PARAMETERS ──────────────────────────────────────────────
BASE_COST_USD   = 150_000.   # annual fixed cost per O&M base
VESSEL_COST_USD = 80_000.    # annual fixed cost per vessel
TECH_COST_USD   = {          # annual fixed cost per technician by skill
    "Trainee":      20_000.,
    "Standard":     30_000.,
    "Professional": 45_000.,
}
# Default technician skill mix proportions
TECH_MIX = {"Trainee": 0.30, "Standard": 0.50, "Professional": 0.20}
TECH_MIX_COST_PER_HEAD = sum(TECH_MIX[s] * TECH_COST_USD[s] for s in TECH_MIX)  # 29,000

# ── VESSEL ROUTING ─────────────────────────────────────────────────────────
VESSEL_SPEED_KPH   = 18.0    # vessel operating speed (km/h used as cost proxy: USD/km)
DIST_BASE_TO_TURB  = 25.0    # base distance to first turbine (km, representative)
DIST_INTER_TURB    = 8.0     # incremental inter-turbine distance (km)
MAX_TURBINES_ROUTE = 5       # maximum turbines per single vessel route

# ── TECHNICIAN UTILISATION (for reporting) ─────────────────────────────────
TECH_UTIL = {
    "Trainee":      {"util_pct": 58, "tasks_per_day": 1.1},
    "Standard":     {"util_pct": 72, "tasks_per_day": 1.7},
    "Professional": {"util_pct": 83, "tasks_per_day": 2.5},
}

# ── RESOURCE CONFIGURATIONS TO EVALUATE ───────────────────────────────────
CONFIGURATIONS = [
    {"label": "Config A (1B/2V/6T)",  "n_bases": 1, "n_vessels": 2, "n_tech": 6},
    {"label": "Config B (1B/3V/9T)",  "n_bases": 1, "n_vessels": 3, "n_tech": 9},
    {"label": "Config C (2B/2V/8T)",  "n_bases": 2, "n_vessels": 2, "n_tech": 8},
    {"label": "Config D (2B/4V/12T)", "n_bases": 2, "n_vessels": 4, "n_tech": 12},
]

# ── ALGORITHM PARAMETERS ───────────────────────────────────────────────────
ROLLING_HORIZON_WINDOW = 7      # forecast window W (days)
MAINT_TRIGGER_R        = 0.62   # reliability threshold to trigger maintenance
OPP_BUNDLE_R           = 0.80   # reliability threshold for opportunistic bundling
ACC_GATE               = 0.45   # minimum accessibility to allow dispatch
LNS_ITERATIONS         = 120    # lower-level LNS iterations
VNS_ITERATIONS         = 50     # upper-level VNS/SA iterations
SA_TEMP_INIT           = 1.0
SA_COOL_RATE           = 0.97
RANDOM_SEED            = 42
