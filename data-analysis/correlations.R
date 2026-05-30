#!/usr/bin/env Rscript
# correlations.R — cross-signal correlation analysis for diagnosing
# why the limb-based herbivores are not learning to walk.
#
# This script joins the per-milestone limb-pool CSV with the main
# population CSV and looks for any signal whatsoever connecting:
#
#   * brain output (limb_target_*, log_std_*) ↔ physical motion
#     (position delta between milestones, velocity_*),
#   * brain value/reward heads (brain_value_v, brain_last_reward,
#     brain_mean_reward_64) ↔ outcomes (predations, distance to prey),
#   * action-clamp rates ↔ exploration parameter drift,
#   * left-vs-right limb symmetry within an organism.
#
# Output: ./results_correlations.txt — one section per question, each
# with the Spearman/Pearson coefficients and a short interpretation.
# Designed to be readable top-to-bottom without running the script.

# ── Paths & milestone discovery ────────────────────────────────────

resolve_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit  <- args[grepl("^--file=", args)]
  if (length(hit) > 0) return(dirname(normalizePath(sub("^--file=", "", hit[1]))))
  normalizePath(".")
}

script_dir   <- resolve_script_dir()
datasets_dir <- normalizePath(file.path(script_dir, "..", "datasets"), mustWork = FALSE)
output_path  <- file.path(script_dir, "results_correlations.txt")

# Mirror general.R / limb_brains.R milestone set so the report stays
# consistent across the suite.
MILESTONES <- list(
  list(label = "0_SECONDS",  secs =     0),
  list(label = "1_MINUTE",   secs =    60),
  list(label = "3_MINUTES",  secs =   180),
  list(label = "5_MINUTES",  secs =   300),
  list(label = "7_MINUTES",  secs =   420),
  list(label = "10_MINUTES", secs =   600),
  list(label = "20_MINUTES", secs =  1200),
  list(label = "30_MINUTES", secs =  1800),
  list(label = "60_MINUTES", secs =  3600)
)

read_semis <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(
    read.csv(path, sep = ";", stringsAsFactors = FALSE, na.strings = c("", "NA")),
    error = function(e) {
      message(sprintf("  skip %s: %s", basename(path), conditionMessage(e)))
      NULL
    }
  )
}

# ── Load all milestones ────────────────────────────────────────────

# Each entry is a list(milestone, secs, main, limb).  `main` is the
# full-population snapshot (one row per organism, every Organism scalar
# + DNA + brain telemetry). `limb` is the per-organism limb-pool
# snapshot (limb_target_0..5, log_std_0..5).
milestone_data <- list()
for (m in MILESTONES) {
  main_path <- file.path(datasets_dir, sprintf("simulation_dataset_%s.csv", m$label))
  limb_path <- file.path(datasets_dir,
                         sprintf("simulation_dataset_%s_limb_herbivore_1.csv", m$label))
  main <- read_semis(main_path)
  limb <- read_semis(limb_path)
  if (is.null(main) || is.null(limb)) next
  milestone_data[[length(milestone_data) + 1]] <- list(
    milestone = m$label, secs = m$secs, main = main, limb = limb
  )
}

if (length(milestone_data) == 0) {
  message(sprintf("No milestone CSVs found in %s", datasets_dir))
  quit(status = 1)
}

# Restrict the main CSV to Level1 limb herbivores for downstream
# correlations. The main dataset reports both photoautotrophs and
# heterotrophs in the same rows; we want only the population whose
# brain rows appear in the limb CSV.
for (i in seq_along(milestone_data)) {
  md <- milestone_data[[i]]
  md$main_h <- subset(md$main,
                      trophic_class == "Herbivore" & intelligence_level == "Level1")
  # Join on entity_id ↔ entity so each organism has both physical
  # state and brain output on the same row.
  md$joined <- merge(md$main_h, md$limb,
                     by.x = "entity_id", by.y = "entity",
                     suffixes = c("", "_limb"))
  milestone_data[[i]] <- md
}

# ── Output sink ────────────────────────────────────────────────────

sink_file <- file(output_path, open = "w")
sink(sink_file, type = "output")
on.exit({ sink(type = "output"); close(sink_file) }, add = TRUE)

hdr <- function(txt, char = "=") {
  cat("\n", strrep(char, 78), "\n", sep = "")
  cat(" ", txt, "\n", sep = "")
  cat(strrep(char, 78), "\n", sep = "")
}
fmt_num <- function(x, digits = 4) {
  if (is.na(x) || !is.finite(x)) return("NA")
  format(round(x, digits), nsmall = digits, scientific = FALSE)
}

# Spearman ρ with sample size + p-value (NA-safe). Returns
# named list(rho, p, n).
spearman <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(list(rho = NA_real_, p = NA_real_, n = sum(ok)))
  ct <- suppressWarnings(cor.test(x[ok], y[ok], method = "spearman", exact = FALSE))
  list(rho = unname(ct$estimate), p = ct$p.value, n = sum(ok))
}
print_corr <- function(label, c) {
  cat(sprintf("  %-50s ρ=%-9s p=%-9s n=%d\n",
              label, fmt_num(c$rho, 3), fmt_num(c$p, 4), c$n))
}

hdr("AEONS Limb-Brain Correlation Diagnostics")
cat("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\n", sep = "")
cat("Milestones loaded: ",
    paste(sapply(milestone_data, function(m) m$milestone), collapse = ", "),
    "\n", sep = "")
cat("Organisms per milestone: ",
    paste(sapply(milestone_data, function(m) nrow(m$joined)), collapse = ", "),
    "\n", sep = "")


# ──────────────────────────────────────────────────────────────────
# Q1. Position drift across milestones — are the bodies moving AT ALL?
# ──────────────────────────────────────────────────────────────────
# The clearest signal: per-organism XZ displacement between adjacent
# milestone snapshots. If the brain has any effect on physics, we
# expect nonzero displacement across 5 → 60 minutes; if displacement
# is exactly zero (to machine precision) across every pair, the brain
# output is not producing motion.

hdr("Q1. Position drift between milestone snapshots", "-")
cat("Per-organism XZ displacement between adjacent milestones.\n")
cat("Δ = |pos_xz(t_b) − pos_xz(t_a)| in world units.\n\n")

# Stitch the limb CSVs across milestones, keyed on entity.
position_long <- do.call(rbind, lapply(milestone_data, function(m) {
  data.frame(
    milestone = m$milestone, secs = m$secs,
    entity = m$limb$entity,
    x = m$limb$x, z = m$limb$z,
    stringsAsFactors = FALSE
  )
}))
entities <- sort(unique(position_long$entity))

# Per-entity displacement table.
disp_rows <- list()
for (e in entities) {
  rows_e <- subset(position_long, entity == e)
  rows_e <- rows_e[order(rows_e$secs), ]
  if (nrow(rows_e) < 2) next
  for (i in 2:nrow(rows_e)) {
    dx <- rows_e$x[i] - rows_e$x[i - 1]
    dz <- rows_e$z[i] - rows_e$z[i - 1]
    disp_rows[[length(disp_rows) + 1]] <- data.frame(
      entity = e,
      from   = rows_e$milestone[i - 1],
      to     = rows_e$milestone[i],
      dt_s   = rows_e$secs[i] - rows_e$secs[i - 1],
      dx     = dx, dz = dz,
      d_xz   = sqrt(dx * dx + dz * dz)
    )
  }
}
disp <- do.call(rbind, disp_rows)
cat("  entity   from→to                    Δt_s     Δx          Δz          |Δ_xz|\n")
cat("  ------   -----------------------    ------   ---------   ---------   --------\n")
for (i in seq_len(nrow(disp))) {
  cat(sprintf("  %-7s  %-9s → %-9s     %6.0f   %9s   %9s   %8s\n",
              disp$entity[i],
              disp$from[i], disp$to[i],
              disp$dt_s[i],
              fmt_num(disp$dx[i], 6),
              fmt_num(disp$dz[i], 6),
              fmt_num(disp$d_xz[i], 6)))
}
cat("\n")
cat("  Summary across all (entity × interval) pairs:\n")
cat(sprintf("    mean |Δ_xz|     = %s\n",   fmt_num(mean(disp$d_xz),     6)))
cat(sprintf("    max  |Δ_xz|     = %s\n",   fmt_num(max(disp$d_xz),      6)))
cat(sprintf("    nonzero count   = %d / %d\n",
            sum(disp$d_xz > 0), nrow(disp)))
if (max(disp$d_xz) == 0) {
  cat("\n  ⚠  EVERY entity is bit-identical across every milestone. Brain\n")
  cat("     output (limb_target_*) varies — see Q2 — but Avian's dynamic\n")
  cat("     bodies are NOT integrating those torques into linear motion.\n")
  cat("     This is the dominant failure mode; the reward shaping and\n")
  cat("     exploration knobs are irrelevant until this is fixed.\n")
}


# ──────────────────────────────────────────────────────────────────
# Q2. Brain output vs physical motion (within milestone-pair)
# ──────────────────────────────────────────────────────────────────
# Does an organism that commands BIGGER limb angles end up moving
# farther between snapshots? If yes, the physics path works and the
# brain just hasn't found the right outputs. If no (and especially if
# Q1 above shows zero motion), the physics path is broken.

hdr("Q2. Brain output amplitude ↔ position drift", "-")

# Aggregate the limb CSV: per (entity × milestone) compute
#   action_norm = mean(|limb_target_*|) across all 6 dims
# Then join consecutive milestones to get an action_norm vs Δ_xz pair.

action_long <- do.call(rbind, lapply(milestone_data, function(m) {
  cols <- sprintf("limb_target_%d", 0:5)
  amp_rows <- sapply(seq_len(nrow(m$limb)), function(i) {
    mean(abs(unlist(m$limb[i, cols])), na.rm = TRUE)
  })
  data.frame(milestone = m$milestone, secs = m$secs,
             entity = m$limb$entity, action_norm = amp_rows,
             stringsAsFactors = FALSE)
}))

# Pair consecutive milestones (entity-aligned) so action_norm[i-1]
# is the brain's "intent" leading up to the displacement from i-1 to i.
pair_rows <- list()
for (e in unique(action_long$entity)) {
  ra <- subset(action_long,   entity == e)
  rd <- subset(disp,          entity == e)
  ra <- ra[order(ra$secs), ]
  rd <- rd[order(rd$dt_s), ]
  if (nrow(ra) < 2 || nrow(rd) < 1) next
  # Pair: ra[i] action_norm with rd[i] displacement (ra[i] is the
  # action at the END of the interval, but ra[i-1] action is the more
  # honest predictor. Both are useful.)
  for (i in seq_len(min(nrow(ra) - 1, nrow(rd)))) {
    pair_rows[[length(pair_rows) + 1]] <- data.frame(
      entity = e,
      action_norm_pre  = ra$action_norm[i],
      action_norm_post = ra$action_norm[i + 1],
      d_xz             = rd$d_xz[i]
    )
  }
}
pairs <- do.call(rbind, pair_rows)
if (!is.null(pairs) && nrow(pairs) > 0) {
  cat("Spearman correlations across (entity × interval) pairs:\n\n")
  print_corr("action_norm at start of interval ↔ Δ_xz",
             spearman(pairs$action_norm_pre,  pairs$d_xz))
  print_corr("action_norm at end of interval ↔ Δ_xz",
             spearman(pairs$action_norm_post, pairs$d_xz))
  if (sd(pairs$d_xz) == 0) {
    cat("\n  ⚠  d_xz has zero variance — correlations are mathematically\n")
    cat("     undefined. This confirms Q1's diagnosis: the issue is not\n")
    cat("     'brain output is too small to move bodies'; the brain output\n")
    cat("     is irrelevant because no output produces motion.\n")
  }
} else {
  cat("  Insufficient (entity × milestone) pairs for analysis.\n")
}


# ──────────────────────────────────────────────────────────────────
# Q3. Cross-organism correlations within each milestone
# ──────────────────────────────────────────────────────────────────
# Does an organism with bigger limb commands also have…
#   * more predations,
#   * lower target_distance,
#   * higher dopamine,
#   * higher brain_value_v?
# Cross-sectional within a milestone — answers "is the brain better
# at producing useful outputs for some organisms than others?"

hdr("Q3. Cross-organism within-milestone correlations", "-")
cat("Per-milestone Spearman correlations across the population at that\n")
cat("snapshot. n is small (one row per organism) so p-values are noisy,\n")
cat("but the direction is what to look at.\n\n")

for (m in milestone_data) {
  d <- m$joined
  if (nrow(d) < 3) {
    cat(sprintf("── %s (n=%d) ── insufficient sample\n", m$milestone, nrow(d)))
    next
  }
  # action norm per organism in the joined frame.
  cols <- sprintf("limb_target_%d", 0:5)
  d$action_norm <- sapply(seq_len(nrow(d)), function(i) mean(abs(unlist(d[i, cols])), na.rm = TRUE))
  cat(sprintf("── %s (n=%d) ──\n", m$milestone, nrow(d)))
  print_corr("action_norm ↔ predations",
             spearman(d$action_norm, d$predations))
  print_corr("action_norm ↔ target_distance",
             spearman(d$action_norm, d$target_distance))
  print_corr("action_norm ↔ dopamine",
             spearman(d$action_norm, d$dopamine))
  if (!is.null(d$brain_value_v)) {
    print_corr("brain_value_v ↔ predations",
               spearman(d$brain_value_v, d$predations))
    print_corr("brain_value_v ↔ target_distance",
               spearman(d$brain_value_v, d$target_distance))
  }
  if (!is.null(d$brain_last_reward)) {
    print_corr("brain_last_reward ↔ predations",
               spearman(d$brain_last_reward, d$predations))
  }
  if (!is.null(d$brain_mean_reward_64)) {
    print_corr("brain_mean_reward_64 ↔ predations",
               spearman(d$brain_mean_reward_64, d$predations))
  }
  cat("\n")
}


# ──────────────────────────────────────────────────────────────────
# Q4. Action-clamp saturation rate
# ──────────────────────────────────────────────────────────────────
# Per-organism fraction of limb-target slots at |value| ≥ 0.99. High
# saturation = brain is screaming at the action clamp. If saturation
# is high AND nothing happens (Q1 = zero motion), the controller is
# the bottleneck. If saturation is low and σ is collapsing, PPO has
# converged into the local minimum.

hdr("Q4. Action-clamp saturation rates", "-")
cat("Per-organism per-milestone fraction of |limb_target_*| ≥ 0.99,\n")
cat("averaged across the 6 action dims.\n\n")

cat("  milestone     entity   sat_rate    action_norm   sigma_mean\n")
cat("  ----------    ------   --------    -----------   ----------\n")
for (m in milestone_data) {
  cols      <- sprintf("limb_target_%d", 0:5)
  sigma_cols <- sprintf("log_std_%d",     0:5)
  for (i in seq_len(nrow(m$limb))) {
    v <- unlist(m$limb[i, cols])
    s <- unlist(m$limb[i, sigma_cols])
    sat <- mean(abs(v) >= 0.99, na.rm = TRUE)
    cat(sprintf("  %-12s  %-7s  %8s    %11s   %10s\n",
                m$milestone, m$limb$entity[i],
                fmt_num(sat, 3),
                fmt_num(mean(abs(v), na.rm = TRUE), 3),
                fmt_num(mean(exp(s), na.rm = TRUE), 3)))
  }
}


# ──────────────────────────────────────────────────────────────────
# Q5. Left/right limb symmetry within an organism
# ──────────────────────────────────────────────────────────────────
# The PD controller in avian_setup pairs limbs (idx 1+2 → pair 0,
# idx 3+4 → pair 1) and flips the X-axis target for the left half.
# But the BRAIN doesn't know about that flip — it outputs one set of
# targets that is shared across both halves of a pair. So per-dim
# correlations within a pair are not meaningful at the brain layer.
# What IS meaningful is whether all six action dims behave the same
# way (signature of uncorrelated outputs = brain has no spatial
# preference yet) or whether some dims are systematically biased
# (signature of a preferred resting pose).

hdr("Q5. Action distribution per dim across the population", "-")
cat("Per-action-dim mean and sd of limb_target values across\n")
cat("(organism × milestone). Mean ≠ 0 → biased policy on that DOF.\n")
cat("Big sd → policy is exploring on that DOF.\n\n")
all_actions <- do.call(rbind, lapply(milestone_data, function(m) {
  data.frame(m$limb[, sprintf("limb_target_%d", 0:5)])
}))
for (d in 0:5) {
  col <- sprintf("limb_target_%d", d)
  v   <- all_actions[[col]]
  cat(sprintf("  dim %d: mean=%-9s sd=%-9s   |mean|/sd=%-9s\n",
              d,
              fmt_num(mean(v, na.rm = TRUE), 3),
              fmt_num(sd(v,   na.rm = TRUE), 3),
              fmt_num(abs(mean(v, na.rm = TRUE)) / sd(v, na.rm = TRUE), 3)))
}


# ──────────────────────────────────────────────────────────────────
# Q6. Log-std (exploration) drift across milestones
# ──────────────────────────────────────────────────────────────────
# log_std lives inside the actor as a trainable parameter. PPO's
# entropy bonus pushes it up, the policy gradient term pulls it down
# as rewarded actions get sharper. Tracking it tells us whether the
# optimizer is doing ANYTHING at all to the policy.

hdr("Q6. Per-dim log_std drift across milestones", "-")
cat("Mean log_std per (milestone × dim). Compare row-over-row:\n")
cat("  * Decreasing → PPO is collapsing exploration (it has found a\n")
cat("    reward signal). For our problem this should still be slow,\n")
cat("    because no organism has ever cashed in any reward.\n")
cat("  * Flat → optimizer is not moving log_std. Either the entropy\n")
cat("    bonus is canceling the policy gradient, or the gradient is\n")
cat("    essentially zero (every policy gets the same return).\n")
cat("  * Increasing → entropy bonus is winning; reward signal too weak\n")
cat("    to commit the policy anywhere.\n\n")

cat("  milestone     dim0       dim1       dim2       dim3       dim4       dim5\n")
cat("  ----------    -------    -------    -------    -------    -------    -------\n")
for (m in milestone_data) {
  cols <- sprintf("log_std_%d", 0:5)
  means <- sapply(cols, function(c) mean(m$limb[[c]], na.rm = TRUE))
  cat(sprintf("  %-12s  %7s    %7s    %7s    %7s    %7s    %7s\n",
              m$milestone,
              fmt_num(means[1], 4),
              fmt_num(means[2], 4),
              fmt_num(means[3], 4),
              fmt_num(means[4], 4),
              fmt_num(means[5], 4),
              fmt_num(means[6], 4)))
}


# ──────────────────────────────────────────────────────────────────
# Q7. Reproductive / cell-growth check
# ──────────────────────────────────────────────────────────────────
# A sanity check that the rest of the simulation is still working —
# the photoautotroph population, organism age, body composition.

hdr("Q7. Population-level sanity", "-")
for (m in milestone_data) {
  main_all <- m$main
  photo_count <- sum(main_all$trophic_class == "Photoautotroph",
                     na.rm = TRUE)
  herb_count  <- sum(main_all$trophic_class == "Herbivore",
                     na.rm = TRUE)
  herbs <- subset(main_all, trophic_class == "Herbivore")
  cat(sprintf("  %s — photos=%d herbs=%d mean_herb_age=%s s mean_grown_cells=%s\n",
              m$milestone, photo_count, herb_count,
              fmt_num(mean(herbs$age_secs, na.rm = TRUE), 1),
              fmt_num(mean(herbs$grown_cell_count, na.rm = TRUE), 1)))
}


cat("\nDone.\n")
