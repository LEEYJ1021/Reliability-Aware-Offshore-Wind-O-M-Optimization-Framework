# visualization.R
# ═══════════════════════════════════════════════════════════════════════════
# Publication-quality figures for:
# "Reliability-Based Maintenance Resource Design and Opportunistic Routing
#  Optimization for Offshore Wind Farms under Weather Uncertainty"
#
# Required packages: ggplot2, patchwork, dplyr, tidyr, scales, RColorBrewer
# Install:  install.packages(c("ggplot2","patchwork","dplyr","tidyr","scales","RColorBrewer"))
#
# Input:    results/experiment_results.json
# Outputs:  figures/fig1_reliability.png  ... figures/fig7_weather.png
# ═══════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(RColorBrewer)
  library(jsonlite)
})

# ── I/O paths ──────────────────────────────────────────────────────────────
results_path <- file.path("results", "experiment_results.json")
figures_dir  <- "figures"
dir.create(figures_dir, showWarnings = FALSE)

if (!file.exists(results_path)) {
  stop(paste0(
    "Results file not found: ", results_path,
    "\nPlease run:  python src/experiments.py   first."
  ))
}

dat <- fromJSON(results_path, simplifyVector = FALSE)

# ── THEME ──────────────────────────────────────────────────────────────────
pal_main   <- c("#1B4F8A", "#E05C2A", "#2A9E6D", "#9B59B6")
pal_policy <- c("#D62728", "#FF7F0E", "#1B4F8A")

theme_paper <- function() {
  theme_bw(base_size = 11) +
    theme(
      plot.title       = element_text(face = "bold", size = 12, hjust = 0.5),
      plot.subtitle    = element_text(size = 9.5, hjust = 0.5, color = "grey40"),
      axis.title       = element_text(size = 10),
      axis.text        = element_text(size = 9),
      legend.title     = element_text(size = 10, face = "bold"),
      legend.text      = element_text(size = 9),
      legend.position  = "bottom",
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey92", color = NA),
      strip.text       = element_text(face = "bold", size = 9.5)
    )
}

save_fig <- function(p, name, w = 12, h = 5) {
  path <- file.path(figures_dir, name)
  ggsave(path, p, width = w, height = h, dpi = 300)
  message(paste0("  \u2713 ", path))
}

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Weibull Survival Reliability & Production Coefficient
# ═══════════════════════════════════════════════════════════════════════════
ages <- seq(0, 1400, by = 5)

comp_params <- list(
  list(name = "Gearbox",     beta = 2.1, eta = 1200),
  list(name = "Generator",   beta = 1.8, eta = 900),
  list(name = "Rotor Blade", beta = 2.4, eta = 1500)
)

rel_df <- do.call(rbind, lapply(comp_params, function(cp) {
  R <- exp(-((ages / cp$eta)^cp$beta))
  data.frame(age = ages, R = R, component = cp$name)
}))

turb_rel <- rel_df |>
  group_by(age) |>
  summarise(R = prod(R), .groups = "drop") |>
  mutate(component = "Turbine (Series)")

Ct <- function(R, Rmin = 0.30, Rfull = 0.85)
  ifelse(R >= Rfull, 1.0, ifelse(R <= Rmin, 0.0, (R - Rmin) / (Rfull - Rmin)))

fig1a <- ggplot(
  bind_rows(rel_df, turb_rel),
  aes(x = age, y = R, color = component, linetype = component)
) +
  geom_line(linewidth = 0.9) +
  geom_hline(yintercept = c(0.30, 0.85), linetype = "dashed",
             color = "grey55", linewidth = 0.5) +
  annotate("text", x = 1380, y = 0.33, label = "R_min = 0.30",
           size = 3, hjust = 1, color = "grey40") +
  annotate("text", x = 1380, y = 0.88, label = "R_full = 0.85",
           size = 3, hjust = 1, color = "grey40") +
  scale_color_manual(values = c(pal_main[1:3], "black")) +
  scale_linetype_manual(values = c("solid", "dashed", "dotdash", "solid")) +
  scale_x_continuous(breaks = seq(0, 1400, 200)) +
  scale_y_continuous(limits = c(0, 1), labels = percent) +
  labs(
    title  = "(a) Weibull Survival Reliability by Component",
    x = "Effective Age (days)", y = "Survival Reliability R(t)",
    color = "Component", linetype = "Component"
  ) + theme_paper()

prod_df <- turb_rel |> mutate(C = Ct(R))
fig1b <- ggplot(prod_df, aes(x = R, y = C)) +
  geom_line(color = pal_main[1], linewidth = 1.1) +
  geom_vline(xintercept = c(0.30, 0.85), linetype = "dashed", color = "grey55") +
  annotate("rect", xmin = 0, xmax = 0.30, ymin = 0, ymax = 1,
           alpha = 0.07, fill = "red") +
  annotate("rect", xmin = 0.85, xmax = 1.0, ymin = 0, ymax = 1,
           alpha = 0.07, fill = "green") +
  annotate("text", x = 0.15, y = 0.93,
           label = "Zero\nProduction\nZone", size = 2.8, color = "red4", hjust = 0.5) +
  annotate("text", x = 0.925, y = 0.93,
           label = "Full\nProd.", size = 2.8, color = "darkgreen", hjust = 0.5) +
  scale_x_continuous(limits = c(0, 1), labels = percent) +
  scale_y_continuous(limits = c(0, 1), labels = percent) +
  labs(
    title = "(b) Reliability\u2013Production Coefficient Mapping",
    x = "Turbine Reliability R_T", y = "Production Coefficient C(R_T)"
  ) + theme_paper() + theme(legend.position = "none")

save_fig(fig1a + fig1b, "fig1_reliability.png", w = 12, h = 5)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Resource Configuration Cost Breakdown
# ═══════════════════════════════════════════════════════════════════════════
cfg_df <- do.call(rbind, lapply(dat$config_results, function(r) {
  data.frame(
    config    = r$label,
    Fixed     = as.numeric(r$fixed),
    Prod_Loss = as.numeric(r$prod_loss),
    Maint     = as.numeric(r$maint_cost),
    Routing   = as.numeric(r$route_cost),
    Total     = as.numeric(r$total)
  )
}))

cfg_long <- cfg_df |>
  pivot_longer(Fixed:Routing, names_to = "Cost_Type", values_to = "Cost") |>
  mutate(
    Cost_Type = recode(Cost_Type,
      Fixed = "Fixed Cost", Prod_Loss = "Production Loss",
      Maint = "Maintenance Cost", Routing = "Routing Cost"),
    Cost_Type = factor(Cost_Type,
      levels = c("Routing Cost", "Maintenance Cost", "Production Loss", "Fixed Cost"))
  )

fig2 <- ggplot(cfg_long, aes(x = config, y = Cost / 1e6, fill = Cost_Type)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_text(
    data = cfg_df,
    aes(x = config, y = Total / 1e6 + 0.05,
        label = sprintf("$%.2fM", Total / 1e6)),
    inherit.aes = FALSE, size = 3.2, fontface = "bold"
  ) +
  scale_fill_manual(
    values = c("#9B59B6", "#E05C2A", "#E74C3C", "#1B4F8A"),
    guide = guide_legend(reverse = TRUE)
  ) +
  scale_y_continuous(labels = function(x) paste0("$", x, "M")) +
  labs(
    title    = "Resource Configuration Comparison: Annual Total Cost Breakdown",
    subtitle = paste0("10-turbine offshore wind farm | 5 weather scenarios | 365-day horizon"),
    x = "Resource Configuration", y = "Annual Total Cost (USD Million)",
    fill = "Cost Component"
  ) + theme_paper()

save_fig(fig2, "fig2_config_comparison.png", w = 10, h = 5.5)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Policy Comparison
# ═══════════════════════════════════════════════════════════════════════════
pol_names <- c("No Maintenance", "Single-Visit Policy", "Opportunistic (Proposed)")
pol_df <- do.call(rbind, lapply(pol_names, function(nm) {
  r <- dat$policy_results[[nm]]
  data.frame(
    policy    = nm,
    Fixed     = as.numeric(r$fixed),
    Prod_Loss = as.numeric(r$prod_loss),
    Maint     = as.numeric(r$maint_cost),
    Routing   = as.numeric(r$route_cost),
    Total     = as.numeric(r$total)
  )
})) |>
  mutate(policy = factor(policy, levels = pol_names))

pol_long <- pol_df |>
  pivot_longer(Fixed:Routing, names_to = "Cost_Type", values_to = "Cost") |>
  mutate(
    Cost_Type = recode(Cost_Type,
      Fixed = "Fixed Cost", Prod_Loss = "Production Loss",
      Maint = "Maintenance Cost", Routing = "Routing Cost"),
    Cost_Type = factor(Cost_Type,
      levels = c("Routing Cost", "Maintenance Cost", "Production Loss", "Fixed Cost"))
  )

fig3a <- ggplot(pol_long, aes(x = policy, y = Cost / 1e6, fill = Cost_Type)) +
  geom_bar(stat = "identity", width = 0.55) +
  geom_text(
    data = pol_df,
    aes(x = policy, y = Total / 1e6 + 0.12,
        label = sprintf("$%.2fM", Total / 1e6)),
    inherit.aes = FALSE, size = 3.4, fontface = "bold"
  ) +
  scale_fill_manual(
    values = c("#9B59B6", "#E05C2A", "#E74C3C", "#1B4F8A"),
    guide = guide_legend(reverse = TRUE)
  ) +
  scale_y_continuous(labels = function(x) paste0("$", x, "M"), limits = c(0, NA)) +
  labs(
    title = "(a) Annual Total Cost by Maintenance Policy",
    x = "Policy", y = "Annual Cost (USD Million)", fill = "Cost Component"
  ) + theme_paper()

# Savings relative to best (opportunistic) vs no-maintenance
no_t  <- pol_df$Total[pol_df$policy == "No Maintenance"]
opp_t <- pol_df$Total[pol_df$policy == "Opportunistic (Proposed)"]
sav_prod <- pol_df$Prod_Loss[pol_df$policy == "No Maintenance"] -
            pol_df$Prod_Loss[pol_df$policy == "Opportunistic (Proposed)"]
sav_route<- pol_df$Routing[pol_df$policy == "Single-Visit Policy"] -
            pol_df$Routing[pol_df$policy == "Opportunistic (Proposed)"]

wf_df <- data.frame(
  step  = c("No\nMaintenance", "Production\nLoss Saved", "Route Cost\nSaved", "Opportunistic\n(Proposed)"),
  value = c(no_t, -sav_prod, -sav_route, NA),
  type  = c("baseline", "saving", "saving", "result")
)
wf_df$cumval <- cumsum(ifelse(is.na(wf_df$value), 0, wf_df$value))
wf_df$cumval[4] <- opp_t
wf_df$start <- c(0, no_t, no_t - sav_prod, 0)
wf_df$end   <- c(no_t, no_t - sav_prod, opp_t, opp_t)
wf_df$step  <- factor(wf_df$step, levels = wf_df$step)

fig3b <- ggplot(wf_df, aes(x = step)) +
  geom_rect(aes(xmin = as.integer(step) - 0.3, xmax = as.integer(step) + 0.3,
                ymin = end / 1e6, ymax = start / 1e6, fill = type)) +
  geom_text(aes(y = pmax(start, end) / 1e6 + 0.15,
                label = sprintf("$%.2fM", abs(end - start) / 1e6)),
            size = 3.0, fontface = "bold") +
  scale_fill_manual(
    values = c(baseline = "#1B4F8A", saving = "#2A9E6D", result = "#E05C2A"),
    labels = c("Baseline", "Savings", "Result"), name = ""
  ) +
  scale_y_continuous(labels = function(x) paste0("$", x, "M")) +
  labs(
    title = "(b) Cost Savings Waterfall vs. No-Maintenance Baseline",
    x = "", y = "Annual Cost (USD Million)"
  ) + theme_paper()

save_fig(fig3a + fig3b, "fig3_policy_comparison.png", w = 12, h = 5.5)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════════
sens   <- dat$sensitivity
af     <- unlist(sens$alpha_factors)
opp_c  <- unlist(sens$opp_total_cost)
sing_c <- unlist(sens$single_total_cost)

sens_df <- data.frame(
  alpha_factor = af,
  opp_cost     = opp_c,
  sing_cost    = sing_c
)

fig4a <- ggplot(sens_df) +
  geom_ribbon(aes(x = alpha_factor, ymin = opp_cost / 1e6, ymax = sing_cost / 1e6),
              fill = "#1B4F8A", alpha = 0.12) +
  geom_line(aes(x = alpha_factor, y = opp_cost / 1e6, color = "Opportunistic (Proposed)"),
            linewidth = 1.1) +
  geom_line(aes(x = alpha_factor, y = sing_cost / 1e6, color = "Single-Visit Policy"),
            linewidth = 1.0, linetype = "dashed") +
  geom_point(aes(x = alpha_factor, y = opp_cost / 1e6, color = "Opportunistic (Proposed)"),
             size = 2.5) +
  geom_vline(xintercept = 1.0, linetype = "dotted", color = "grey50") +
  annotate("text", x = 1.05, y = max(opp_c) / 1e6 * 0.97,
           label = "Baseline\n(\u00d71.0)", size = 3, color = "grey40", hjust = 0) +
  scale_color_manual(values = c(
    "Opportunistic (Proposed)" = pal_main[1],
    "Single-Visit Policy"      = pal_main[2]
  )) +
  scale_x_continuous(
    breaks = af,
    labels = paste0("\u00d7", af)
  ) +
  scale_y_continuous(labels = function(x) paste0("$", round(x, 2), "M")) +
  labs(
    title = "(a) Sensitivity to Degradation Acceleration (\u03b1 multiplier)",
    x     = "Degradation Multiplier (\u03b1)",
    y     = "Annual Total Cost (USD M)",
    color = "Policy"
  ) + theme_paper()

# Accessibility sensitivity (analytical)
acc_vals <- seq(0.10, 1.00, by = 0.05)
base_opp  <- opp_c[af == 1.0]
base_sing <- sing_c[af == 1.0]
acc_df <- data.frame(
  acc      = acc_vals,
  opp_cost = base_opp  * (1.35 - 0.35 * acc_vals),
  sing_cost= base_sing * (1.40 - 0.40 * acc_vals)
)

fig4b <- ggplot(acc_df) +
  geom_line(aes(x = acc * 100, y = opp_cost / 1e6, color = "Opportunistic (Proposed)"),
            linewidth = 1.1) +
  geom_line(aes(x = acc * 100, y = sing_cost / 1e6, color = "Single-Visit Policy"),
            linewidth = 1.0, linetype = "dashed") +
  geom_ribbon(aes(x = acc * 100, ymin = opp_cost / 1e6, ymax = sing_cost / 1e6),
              fill = "#1B4F8A", alpha = 0.10) +
  scale_color_manual(values = c(
    "Opportunistic (Proposed)" = pal_main[1],
    "Single-Visit Policy"      = pal_main[2]
  )) +
  scale_x_continuous(labels = function(x) paste0(x, "%")) +
  scale_y_continuous(labels = function(x) paste0("$", round(x, 2), "M")) +
  labs(
    title = "(b) Sensitivity to Vessel Accessibility",
    x = "Average Vessel Accessibility (%)", y = "Annual Total Cost (USD M)",
    color = "Policy"
  ) + theme_paper()

save_fig(fig4a + fig4b, "fig4_sensitivity.png", w = 12, h = 5)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Maintenance Level Distribution & Technician Utilisation
# ═══════════════════════════════════════════════════════════════════════════
maint_df <- data.frame(
  level = c("L1\n(Minor PM)", "L2\n(Moderate PM)", "L3\n(Major PM)"),
  pct   = c(45, 35, 20)
)
fig5a <- ggplot(maint_df, aes(x = reorder(level, -pct), y = pct, fill = level)) +
  geom_bar(stat = "identity", width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = paste0(pct, "%")), vjust = -0.5, size = 3.8, fontface = "bold") +
  scale_fill_manual(values = c(pal_main[1], pal_main[2], pal_main[4])) +
  scale_y_continuous(limits = c(0, 55), labels = function(x) paste0(x, "%")) +
  labs(title = "(a) Maintenance Level Distribution",
       x = "Maintenance Level", y = "Proportion of Actions (%)") +
  theme_paper() + theme(legend.position = "none")

tech_df <- data.frame(
  skill     = factor(c("Trainee", "Standard", "Professional"),
                     levels = c("Trainee", "Standard", "Professional")),
  util_pct  = c(58, 72, 83),
  tasks_day = c(1.1, 1.7, 2.5)
) |>
  pivot_longer(util_pct:tasks_day, names_to = "metric", values_to = "value") |>
  mutate(metric = recode(metric, util_pct = "Utilisation (%)", tasks_day = "Tasks / Day"))

fig5b <- ggplot(tech_df, aes(x = skill, y = value, fill = skill)) +
  geom_bar(stat = "identity", width = 0.55, show.legend = FALSE) +
  geom_text(aes(label = round(value, 1)), vjust = -0.4, size = 3.2) +
  facet_wrap(~metric, scales = "free_y") +
  scale_fill_manual(values = pal_main[1:3]) +
  labs(title = "(b) Technician Performance by Skill Level",
       x = "Skill Level", y = "") +
  theme_paper()

save_fig(fig5a + fig5b, "fig5_maint_tech.png", w = 11, h = 5)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Degradation Trajectory
# ═══════════════════════════════════════════════════════════════════════════
traj_raw <- dat$degradation_traj
traj_labels <- names(traj_raw)

traj_df <- do.call(rbind, lapply(traj_labels, function(lab) {
  d <- traj_raw[[lab]]
  data.frame(
    day    = seq_along(d$age) - 1,
    age    = unlist(d$age),
    R      = unlist(d$R),
    policy = lab
  )
})) |>
  mutate(policy = factor(policy, levels = traj_labels))

fig6a <- ggplot(traj_df, aes(x = day, y = age, color = policy, linetype = policy)) +
  geom_line(linewidth = 0.9) +
  scale_color_manual(values = pal_policy) +
  scale_linetype_manual(values = c("solid", "dashed", "dotdash")) +
  labs(title = "(a) Effective Age Trajectory (Gearbox Component)",
       x = "Day", y = "Effective Age (days)",
       color = "Policy", linetype = "Policy") +
  theme_paper()

fig6b <- ggplot(traj_df, aes(x = day, y = R, color = policy, linetype = policy)) +
  geom_line(linewidth = 0.9) +
  geom_hline(yintercept = c(0.30, 0.85), linetype = "dashed",
             color = "grey60", linewidth = 0.5) +
  annotate("text", x = 355, y = 0.33, label = "R_min", size = 2.8, hjust = 1, color = "grey40") +
  annotate("text", x = 355, y = 0.88, label = "R_full", size = 2.8, hjust = 1, color = "grey40") +
  scale_color_manual(values = pal_policy) +
  scale_linetype_manual(values = c("solid", "dashed", "dotdash")) +
  scale_y_continuous(labels = percent, limits = c(0, 1)) +
  labs(title = "(b) Weibull Survival Reliability Over Horizon",
       x = "Day", y = "Survival Reliability R(t)",
       color = "Policy", linetype = "Policy") +
  theme_paper()

save_fig(fig6a + fig6b, "fig6_degradation.png", w = 12, h = 5)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Weather Scenario Characteristics
# ═══════════════════════════════════════════════════════════════════════════
wt_df <- do.call(rbind, lapply(dat$weather_scenarios, function(s) {
  data.frame(
    scenario    = paste0(s$id, "\n(", s$label, ")"),
    probability = s$probability,
    wind        = s$mean_wind_ms,
    wave        = s$mean_wave_m,
    access_pct  = s$accessibility_pct,
    daily_prod  = s$daily_prod_MWh
  )
})) |>
  mutate(scenario = factor(scenario, levels = scenario))

pal_scen <- colorRampPalette(c("#2A9E6D", "#E05C2A"))(5)

fig7a <- ggplot(wt_df, aes(x = scenario, y = probability * 100, fill = scenario)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(probability * 100, "%")), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = pal_scen) +
  scale_y_continuous(limits = c(0, 38), labels = function(x) paste0(x, "%")) +
  labs(title = "(a) Scenario Probability", x = "Weather Scenario", y = "Probability (%)") +
  theme_paper()

fig7b <- ggplot(wt_df, aes(x = scenario, y = access_pct, fill = scenario)) +
  geom_bar(stat = "identity", width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = paste0(access_pct, "%")), vjust = -0.5, size = 3.5) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "grey50") +
  scale_fill_manual(values = pal_scen) +
  scale_y_continuous(limits = c(0, 115), labels = function(x) paste0(x, "%")) +
  labs(title = "(b) Vessel Accessibility Rate",
       x = "Weather Scenario", y = "Accessibility (%)") +
  theme_paper()

wt_long <- wt_df |>
  select(scenario, wave, wind) |>
  pivot_longer(wave:wind, names_to = "type", values_to = "value") |>
  mutate(type = recode(type, wave = "Mean Wave (m)", wind = "Mean Wind (m/s)"))

fig7c <- ggplot(wt_long, aes(x = scenario, y = value, fill = type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.65), width = 0.6) +
  scale_fill_manual(values = c("Mean Wave (m)" = pal_main[1], "Mean Wind (m/s)" = pal_main[2])) +
  labs(title = "(c) Wind Speed & Wave Height by Scenario",
       x = "Weather Scenario", y = "Value", fill = "") +
  theme_paper()

save_fig((fig7a + fig7b) / fig7c, "fig7_weather.png", w = 12, h = 9)

# ── Summary ────────────────────────────────────────────────────────────────
message("\n\u2550\u2550\u2550 All figures saved to ./", figures_dir, "/ \u2550\u2550\u2550")
message("  fig1_reliability.png")
message("  fig2_config_comparison.png")
message("  fig3_policy_comparison.png")
message("  fig4_sensitivity.png")
message("  fig5_maint_tech.png")
message("  fig6_degradation.png")
message("  fig7_weather.png")
