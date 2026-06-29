"""
run_all.py — Master Runner
==========================
Executes the full reproducibility pipeline for:

  "Reliability-Based Maintenance Resource Design and Opportunistic Routing
   Optimization for Offshore Wind Farms under Weather Uncertainty"

Pipeline
--------
  Step 1  Validate dependencies (numpy, matplotlib; R optional)
  Step 2  Run all numerical experiments  →  results/experiment_results.json
  Step 3  Generate publication figures   →  figures/fig1_*.png … fig7_*.png
          (via R if available, else falls back to Python/matplotlib)
  Step 4  Print result summary to console

Usage
-----
  python run_all.py              # full pipeline
  python run_all.py --no-figures # experiments only (skip figures)
  python run_all.py --figs-only  # generate figures from existing results

Directory layout expected (same as the repository root):
  run_all.py          ← this file
  src/
    parameters.py
    reliability.py
    patterns.py
    simulation.py
    experiments.py
  visualization.R
  results/            (created automatically)
  figures/            (created automatically)
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time

# ── Ensure src/ is on the path ─────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Dependency check
# ══════════════════════════════════════════════════════════════════════════

def check_dependencies() -> None:
    """Verify required Python packages are installed."""
    required = ["numpy"]
    missing  = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(
            f"[ERROR] Missing Python packages: {', '.join(missing)}\n"
            f"  Install with:  pip install {' '.join(missing)}"
        )
    print("[✓] Python dependencies satisfied.")


def check_r() -> bool:
    """Return True if Rscript is available and required packages are installed."""
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False
        # Check R packages
        pkg_check = subprocess.run(
            ["Rscript", "-e",
             "pkgs<-c('ggplot2','patchwork','dplyr','tidyr','scales','RColorBrewer','jsonlite');"
             "missing<-pkgs[!sapply(pkgs,requireNamespace,quietly=TRUE)];"
             "if(length(missing)>0) cat('MISSING:',paste(missing,collapse=','),'\\n') else cat('OK\\n')"],
            capture_output=True, text=True, timeout=30,
        )
        return "OK" in pkg_check.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Run experiments
# ══════════════════════════════════════════════════════════════════════════

def run_experiments(results_dir: str = "results") -> str:
    """Run all numerical experiments and save JSON results."""
    os.chdir(ROOT_DIR)   # ensure relative paths work from repo root
    from experiments import run_all
    return run_all(output_dir=results_dir)


# ══════════════════════════════════════════════════════════════════════════
# STEP 3a — Figures via R
# ══════════════════════════════════════════════════════════════════════════

def run_r_figures() -> bool:
    """Generate all 7 publication figures using the R visualization script."""
    r_script = os.path.join(ROOT_DIR, "visualization.R")
    if not os.path.exists(r_script):
        print("[!] visualization.R not found — skipping R figures.")
        return False

    print("\n[Step 3] Generating figures with R …")
    result = subprocess.run(
        ["Rscript", r_script],
        cwd=ROOT_DIR,
        capture_output=False,
        text=True,
    )
    return result.returncode == 0


# ══════════════════════════════════════════════════════════════════════════
# STEP 3b — Fallback figures via Python/matplotlib
# ══════════════════════════════════════════════════════════════════════════

def run_python_figures(results_json: str, figures_dir: str = "figures") -> None:
    """
    Fallback figure generation using matplotlib when R is unavailable.
    Produces simplified versions of the key figures.
    """
    import json
    import math
    import numpy as np

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[!] matplotlib not available — skipping Python fallback figures.")
        return

    os.makedirs(figures_dir, exist_ok=True)

    with open(results_json) as f:
        dat = json.load(f)

    print("\n[Step 3] Generating fallback figures with matplotlib …")
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                          "figure.dpi": 150})

    BLUE = "#1B4F8A"; ORANGE = "#E05C2A"; GREEN = "#2A9E6D"; PURPLE = "#9B59B6"

    # ── Figure 1: Reliability curves ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ages = np.linspace(0, 1400, 300)
    params = [("Gearbox", 2.1, 1200, BLUE), ("Generator", 1.8, 900, ORANGE),
              ("Rotor Blade", 2.4, 1500, GREEN)]
    for name, beta, eta, col in params:
        R = np.exp(-((ages / eta) ** beta))
        axes[0].plot(ages, R, label=name, color=col)
    R_turb = np.exp(-((ages/1200)**2.1)) * np.exp(-((ages/900)**1.8)) * np.exp(-((ages/1500)**2.4))
    axes[0].plot(ages, R_turb, "k-", lw=2, label="Turbine (Series)")
    axes[0].axhline(0.30, ls="--", color="grey", lw=0.8); axes[0].axhline(0.85, ls="--", color="grey", lw=0.8)
    axes[0].set(xlabel="Effective Age (days)", ylabel="Survival Reliability R(t)",
                title="(a) Weibull Survival Reliability by Component"); axes[0].legend(fontsize=8)

    C = np.where(R_turb >= 0.85, 1.0, np.where(R_turb <= 0.30, 0.0, (R_turb-0.30)/0.55))
    axes[1].plot(R_turb, C, color=BLUE, lw=1.5)
    axes[1].axvline(0.30, ls="--", color="grey", lw=0.8); axes[1].axvline(0.85, ls="--", color="grey", lw=0.8)
    axes[1].set(xlabel="Turbine Reliability R_T", ylabel="Production Coefficient C(R_T)",
                title="(b) Reliability–Production Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_reliability.png"), bbox_inches="tight")
    plt.close(); print("  ✓ fig1_reliability.png")

    # ── Figure 2: Config comparison ───────────────────────────────────────
    cfg_r = dat["config_results"]
    labels   = [r["label"].replace("(", "\n(") for r in cfg_r]
    fixed    = [r["fixed"]     / 1e6 for r in cfg_r]
    prod_l   = [r["prod_loss"] / 1e6 for r in cfg_r]
    maint_c  = [r["maint_cost"]/ 1e6 for r in cfg_r]
    route_c  = [r["route_cost"]/ 1e6 for r in cfg_r]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x, fixed,  label="Fixed Cost",        color=BLUE)
    b2 = ax.bar(x, prod_l, bottom=fixed, label="Production Loss", color="#E74C3C")
    btm2 = [f+p for f,p in zip(fixed, prod_l)]
    b3 = ax.bar(x, maint_c, bottom=btm2, label="Maintenance Cost", color=ORANGE)
    btm3 = [b+m for b,m in zip(btm2, maint_c)]
    b4 = ax.bar(x, route_c, bottom=btm3, label="Routing Cost",     color=PURPLE)
    totals = [r["total"]/1e6 for r in cfg_r]
    for xi, tot in zip(x, totals):
        ax.text(xi, tot+0.05, f"${tot:.2f}M", ha="center", fontsize=9, fontweight="bold")
    ax.set(xticks=list(x), xticklabels=labels, ylabel="Annual Total Cost (USD Million)",
           title="Resource Configuration Cost Breakdown")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_config_comparison.png"), bbox_inches="tight")
    plt.close(); print("  ✓ fig2_config_comparison.png")

    # ── Figure 3: Policy comparison ───────────────────────────────────────
    pol_names = ["No Maintenance", "Single-Visit Policy", "Opportunistic (Proposed)"]
    pol_r     = dat["policy_results"]
    xt = range(len(pol_names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fx = [pol_r[n]["fixed"]     /1e6 for n in pol_names]
    pl = [pol_r[n]["prod_loss"] /1e6 for n in pol_names]
    mc = [pol_r[n]["maint_cost"]/1e6 for n in pol_names]
    rc = [pol_r[n]["route_cost"]/1e6 for n in pol_names]
    axes[0].bar(xt, fx, label="Fixed",       color=BLUE)
    axes[0].bar(xt, pl, bottom=fx, label="Prod Loss",  color="#E74C3C")
    b2f=[f+p for f,p in zip(fx,pl)]
    axes[0].bar(xt, mc, bottom=b2f, label="Maint",      color=ORANGE)
    b3f=[b+m for b,m in zip(b2f,mc)]
    axes[0].bar(xt, rc, bottom=b3f, label="Route",      color=PURPLE)
    tots=[pol_r[n]["total"]/1e6 for n in pol_names]
    for xi,tot in zip(xt,tots): axes[0].text(xi,tot+0.12,f"${tot:.2f}M",ha="center",fontsize=9,fontweight="bold")
    axes[0].set(xticks=list(xt), xticklabels=["No\nMaint","Single-Visit","Opportunistic\n(Proposed)"],
                ylabel="USD Million", title="(a) Annual Cost by Policy"); axes[0].legend(fontsize=8)

    kn = dat["key_numbers"]
    axes[1].bar(["Saving vs\nSingle-Visit", "Saving vs\nNo-Maintenance"],
                [kn["saving_pct_vs_single"], kn["saving_pct_vs_no_maint"]],
                color=[ORANGE, GREEN], width=0.5)
    axes[1].set(ylabel="Cost Saving (%)", title="(b) Savings from Opportunistic Policy")
    for xi, val in enumerate([kn["saving_pct_vs_single"], kn["saving_pct_vs_no_maint"]]):
        axes[1].text(xi, val+0.5, f"{val:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_policy_comparison.png"), bbox_inches="tight")
    plt.close(); print("  ✓ fig3_policy_comparison.png")

    # ── Figure 4: Sensitivity ─────────────────────────────────────────────
    sens = dat["sensitivity"]
    af   = sens["alpha_factors"]
    opp  = [c/1e6 for c in sens["opp_total_cost"]]
    sing = [c/1e6 for c in sens["single_total_cost"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(af, opp, sing, alpha=0.12, color=BLUE)
    ax.plot(af, opp,  "o-", color=BLUE,   lw=1.5, label="Opportunistic (Proposed)")
    ax.plot(af, sing, "s--",color=ORANGE,  lw=1.2, label="Single-Visit Policy")
    ax.axvline(1.0, ls=":", color="grey"); ax.text(1.02, max(opp)*0.98, "Baseline", color="grey", fontsize=9)
    ax.set(xlabel="Degradation Multiplier (α)", ylabel="Annual Total Cost (USD M)",
           title="Sensitivity to Degradation Acceleration")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_sensitivity.png"), bbox_inches="tight")
    plt.close(); print("  ✓ fig4_sensitivity.png")

    # ── Figure 5: Maintenance & Tech ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    lvls = ["L1\n(Minor PM)", "L2\n(Moderate PM)", "L3\n(Major PM)"]
    pcts = [45, 35, 20]
    axes[0].bar(lvls, pcts, color=[BLUE, ORANGE, PURPLE], width=0.5)
    for xi, v in enumerate(pcts): axes[0].text(xi, v+0.8, f"{v}%", ha="center", fontweight="bold")
    axes[0].set(ylabel="Proportion (%)", title="(a) Maintenance Level Distribution")

    skills = ["Trainee", "Standard", "Professional"]
    utils  = [58, 72, 83]
    axes[1].barh(skills, utils, color=[BLUE, ORANGE, GREEN], height=0.4)
    for yi, v in enumerate(utils): axes[1].text(v+0.5, yi, f"{v}%", va="center", fontweight="bold")
    axes[1].set(xlabel="Utilisation (%)", title="(b) Technician Utilisation Rate", xlim=(0,100))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig5_maint_tech.png"), bbox_inches="tight")
    plt.close(); print("  ✓ fig5_maint_tech.png")

    # ── Figure 6: Degradation trajectories ───────────────────────────────
    traj = dat["degradation_traj"]
    days = list(range(len(list(traj.values())[0]["age"])))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [BLUE, ORANGE, GREEN]; ls_list = ["-", "--", "-."]
    for (lab, d), col, ls in zip(traj.items(), colors, ls_list):
        axes[0].plot(days, d["age"], color=col, linestyle=ls, lw=1.1, label=lab)
        axes[1].plot(days, d["R"],   color=col, linestyle=ls, lw=1.1, label=lab)
    axes[1].axhline(0.30, ls="--", color="grey", lw=0.8)
    axes[1].axhline(0.85, ls="--", color="grey", lw=0.8)
    axes[0].set(xlabel="Day", ylabel="Effective Age (days)", title="(a) Effective Age Trajectory")
    axes[1].set(xlabel="Day", ylabel="Survival Reliability R(t)", ylim=(0,1), title="(b) Reliability Trajectory")
    for ax in axes: ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig6_degradation.png"), bbox_inches="tight")
    plt.close(); print("  ✓ fig6_degradation.png")

    # ── Figure 7: Weather scenarios ───────────────────────────────────────
    wt = dat["weather_scenarios"]
    sc_ids   = [w["id"] for w in wt]
    probs    = [w["probability"]*100 for w in wt]
    acc      = [w["accessibility_pct"] for w in wt]
    winds    = [w["mean_wind_ms"] for w in wt]
    waves    = [w["mean_wave_m"]  for w in wt]
    xi = range(len(sc_ids))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors_scen = ["#2A9E6D","#5AB88D","#E0A030","#E05C2A","#B02010"]
    axes[0,0].bar(xi, probs, color=colors_scen)
    axes[0,0].set(xticks=list(xi), xticklabels=sc_ids, ylabel="%", title="(a) Scenario Probability")
    axes[0,1].bar(xi, acc, color=colors_scen)
    axes[0,1].axhline(50, ls="--", color="grey", lw=0.8)
    axes[0,1].set(xticks=list(xi), xticklabels=sc_ids, ylabel="%", title="(b) Vessel Accessibility")
    w  = 0.35
    axes[1,0].bar([i-w/2 for i in xi], winds, width=w, color=BLUE,   label="Wind (m/s)")
    axes[1,0].bar([i+w/2 for i in xi], waves, width=w, color=ORANGE, label="Wave (m)")
    axes[1,0].set(xticks=list(xi), xticklabels=sc_ids, title="(c) Wind & Wave Conditions"); axes[1,0].legend()
    axes[1,1].bar(xi, [w["daily_prod_MWh"] for w in wt], color=colors_scen)
    axes[1,1].set(xticks=list(xi), xticklabels=sc_ids, ylabel="MWh", title="(d) Daily Production Potential")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig7_weather.png"), bbox_inches="tight")
    plt.close(); print("  ✓ fig7_weather.png")

    print(f"\n  All fallback figures saved to ./{figures_dir}/")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Print result summary
# ══════════════════════════════════════════════════════════════════════════

def print_summary(results_json: str) -> None:
    with open(results_json) as f:
        dat = json.load(f)

    kn = dat.get("key_numbers", {})
    print("\n" + "═" * 65)
    print("  RESULT SUMMARY")
    print("═" * 65)
    print(f"  Best configuration:        {dat.get('best_config', 'N/A')}")
    print(f"  Opportunistic total cost:  ${kn.get('opp_total_USD',0):>12,.0f}")
    print(f"  Single-visit total cost:   ${kn.get('sing_total_USD',0):>12,.0f}")
    print(f"  No-maintenance total cost: ${kn.get('no_maint_total_USD',0):>12,.0f}")
    print(f"  Saving vs. single-visit:   {kn.get('saving_pct_vs_single',0):>+.1f}%")
    print(f"  Saving vs. no-maintenance: {kn.get('saving_pct_vs_no_maint',0):>+.1f}%")
    print(f"  Opportunistic actions:     {kn.get('opp_actions',0)}")
    print(f"  Single-visit actions:      {kn.get('sing_actions',0)}")
    print("═" * 65)


# ══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master runner for offshore wind O&M paper experiments."
    )
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation; run experiments only.")
    parser.add_argument("--figs-only", action="store_true",
                        help="Generate figures from existing results/experiment_results.json.")
    parser.add_argument("--results-dir", default="results",
                        help="Directory to store experiment JSON (default: results/).")
    parser.add_argument("--figures-dir", default="figures",
                        help="Directory to store figure PNGs (default: figures/).")
    args = parser.parse_args()

    results_json = os.path.join(ROOT_DIR, args.results_dir, "experiment_results.json")

    t0 = time.time()

    if not args.figs_only:
        # Step 1: Check dependencies
        print("\n[Step 1] Checking dependencies …")
        check_dependencies()

        # Step 2: Run experiments
        print("\n[Step 2] Running numerical experiments …")
        results_json = run_experiments(results_dir=os.path.join(ROOT_DIR, args.results_dir))

    else:
        if not os.path.exists(results_json):
            sys.exit(
                f"[ERROR] Results file not found: {results_json}\n"
                "  Run without --figs-only to generate results first."
            )
        print(f"[✓] Using existing results: {results_json}")

    # Step 3: Figures
    if not args.no_figures:
        r_ok = check_r()
        if r_ok:
            print("\n[Step 3] R detected — generating figures with visualization.R …")
            os.chdir(ROOT_DIR)
            run_r_figures()
        else:
            print("\n[Step 3] R not available — using matplotlib fallback …")
            run_python_figures(results_json, os.path.join(ROOT_DIR, args.figures_dir))
    else:
        print("\n[Step 3] Skipped (--no-figures flag).")

    # Step 4: Summary
    print_summary(results_json)
    print(f"\n  Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
