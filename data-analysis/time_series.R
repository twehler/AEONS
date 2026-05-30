#!/usr/bin/env Rscript
# time_series.R — within-run learning-trajectory analysis.
#
# Reads the continuous per-organism trace written by
# `time_series_log.rs` — one row per Level1 SLIDING herbivore every
# `LOG_INTERVAL_SECS` (5) virtual seconds, now landing in
# `../datasets/time_series_<timestamp>.csv`. Unlike the milestone
# snapshots (coarse point-in-time samples), this is a dense time
# series, so it can show whether the population is actually *learning*
# over the course of a run — reward trending up, target distance
# trending down, etc.
#
# Picks the MOST RECENT trace file by modification time (each run
# writes its own timestamped file). Output:
# ./results_time_series.txt
#
# Usage:
#   cd data-analysis && Rscript time_series.R
# or
#   Rscript data-analysis/time_series.R

# ── Paths ──────────────────────────────────────────────────────────

resolve_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit  <- args[grepl("^--file=", args)]
  if (length(hit) > 0) return(dirname(normalizePath(sub("^--file=", "", hit[1]))))
  normalizePath(".")
}

script_dir   <- resolve_script_dir()
datasets_dir <- normalizePath(file.path(script_dir, "..", "datasets"), mustWork = FALSE)
output_path  <- file.path(script_dir, "results_time_series.txt")

# Discover all time-series traces, pick the newest by mtime.
candidates <- list.files(datasets_dir, pattern = "^time_series_.*\\.csv$",
                         full.names = TRUE)
if (length(candidates) == 0) {
  message(sprintf("No time_series_*.csv found in %s", datasets_dir))
  message("Run the simulation for at least a few seconds (the logger fires ",
          "every 5 virtual seconds while running).")
  quit(status = 1)
}
info  <- file.info(candidates)
trace <- rownames(info)[which.max(info$mtime)]

df <- tryCatch(
  read.csv(trace, sep = ";", stringsAsFactors = FALSE, na.strings = c("", "NA")),
  error = function(e) { message(sprintf("read failed: %s", conditionMessage(e))); NULL }
)
if (is.null(df) || nrow(df) == 0) {
  message("Trace file is empty.")
  quit(status = 1)
}

# ── Output sink + helpers ──────────────────────────────────────────

sink_file <- file(output_path, open = "w")
sink(sink_file, type = "output")
on.exit({ sink(type = "output"); close(sink_file) }, add = TRUE)

hdr <- function(txt, char = "=") {
  cat("\n", strrep(char, 78), "\n", sep = "")
  cat(" ", txt, "\n", sep = "")
  cat(strrep(char, 78), "\n", sep = "")
}
fmt <- function(x, d = 4) {
  if (length(x) == 0 || is.na(x) || !is.finite(x)) return("NA")
  format(round(x, d), nsmall = d, scientific = FALSE)
}
spearman <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(list(rho = NA_real_, p = NA_real_, n = sum(ok)))
  ct <- suppressWarnings(cor.test(x[ok], y[ok], method = "spearman", exact = FALSE))
  list(rho = unname(ct$estimate), p = ct$p.value, n = sum(ok))
}
# OLS slope of y ~ t with p-value.
trend <- function(t, y) {
  ok <- is.finite(t) & is.finite(y)
  if (sum(ok) < 3 || length(unique(t[ok])) < 2) {
    return(list(slope = NA_real_, p = NA_real_, r2 = NA_real_))
  }
  fit <- lm(y[ok] ~ t[ok])
  s   <- summary(fit)
  list(slope = unname(coef(fit)[2]),
       p     = if (nrow(s$coefficients) >= 2) s$coefficients[2, 4] else NA_real_,
       r2    = s$r.squared)
}

# ── Overview ───────────────────────────────────────────────────────

hdr("Within-Run Learning Trajectory (sliding Level1 herbivores)")
cat("Generated:  ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\n", sep = "")
cat("Trace file: ", basename(trace), "\n", sep = "")
cat("Rows:       ", nrow(df), "\n", sep = "")
cat("Organisms:  ", length(unique(df$entity_id)), " distinct entity_ids\n", sep = "")
cat("Time span:  ", fmt(min(df$virtual_time_secs), 1), " → ",
    fmt(max(df$virtual_time_secs), 1), " virtual seconds\n", sep = "")

# ── Time-binned aggregate trends ───────────────────────────────────
# Bin the run into ~20 equal windows and report the population mean of
# the key learning signals in each. The shape of these columns is the
# headline result: is reward climbing, is target distance shrinking?

hdr("Population means over time (binned)", "-")

t_min <- min(df$virtual_time_secs); t_max <- max(df$virtual_time_secs)
n_bins <- 20
span   <- max(t_max - t_min, 1e-6)
df$bin <- pmin(n_bins, 1 + floor((df$virtual_time_secs - t_min) / span * n_bins))

agg_cols <- c("brain_mean_reward_64", "brain_value_v", "predations",
              "target_distance", "movement_speed", "dopamine",
              "brain_last_eat_component", "brain_last_progress_component")
agg_cols <- agg_cols[agg_cols %in% names(df)]

cat(sprintf("  %-8s %8s", "t_mid_s", "n"))
for (c in agg_cols) cat(sprintf(" %18s", c))
cat("\n")
cat("  ", strrep("-", 8 + 8 + 19 * length(agg_cols)), "\n", sep = "")

for (b in sort(unique(df$bin))) {
  sub <- df[df$bin == b, , drop = FALSE]
  t_mid <- t_min + (b - 0.5) / n_bins * span
  cat(sprintf("  %8s %8d", fmt(t_mid, 0), nrow(sub)))
  for (c in agg_cols) cat(sprintf(" %18s", fmt(mean(sub[[c]], na.rm = TRUE), 4)))
  cat("\n")
}

# ── Trend tests across the whole run ───────────────────────────────
# Per-row OLS against virtual time. A positive, significant slope on
# `brain_mean_reward_64` is the cleanest "the policy is learning"
# signal; a negative slope on `target_distance` means herbivores are
# getting closer to prey over time.

hdr("Linear trend vs virtual time (whole run)", "-")
cat("  signal                          slope/sec        p           R^2\n")
cat("  ----------------------------  ------------   ----------   --------\n")
for (c in agg_cols) {
  tr <- trend(df$virtual_time_secs, df[[c]])
  cat(sprintf("  %-28s  %12s   %10s   %8s\n",
              c, fmt(tr$slope, 6), fmt(tr$p, 4), fmt(tr$r2, 4)))
}
cat("\n  Read: +slope on reward / value / predations = learning;\n")
cat("        -slope on target_distance = closing on prey over the run.\n")

# ── Cross-signal correlations ──────────────────────────────────────

hdr("Cross-signal correlations (per-row Spearman)", "-")
pairs <- list(
  c("target_distance",     "brain_mean_reward_64"),
  c("movement_speed",      "predations"),
  c("brain_value_v",       "brain_mean_reward_64"),
  c("target_distance",     "movement_speed"),
  c("dopamine",            "predations")
)
for (p in pairs) {
  if (all(p %in% names(df))) {
    c <- spearman(df[[p[1]]], df[[p[2]]])
    cat(sprintf("  %-26s <-> %-26s  rho=%-8s p=%-8s n=%d\n",
                p[1], p[2], fmt(c$rho, 3), fmt(c$p, 4), c$n))
  }
}

# ── Per-organism endpoints: best vs worst learners ─────────────────
# For each organism, take its LAST logged row and rank by
# mean_reward_64. Shows whether learning is broad or carried by a few.

hdr("Per-organism final state (top / bottom by mean_reward_64)", "-")
last_rows <- do.call(rbind, lapply(split(df, df$entity_id), function(g) {
  g[which.max(g$virtual_time_secs), , drop = FALSE]
}))
last_rows <- last_rows[order(-last_rows$brain_mean_reward_64), , drop = FALSE]

show_n <- min(8, nrow(last_rows))
report_rows <- function(rows) {
  cat(sprintf("  %-10s %12s %10s %14s %12s\n",
              "entity", "mean_rew_64", "predations", "target_dist", "value_v"))
  for (i in seq_len(nrow(rows))) {
    cat(sprintf("  %-10s %12s %10s %14s %12s\n",
                rows$entity_id[i],
                fmt(rows$brain_mean_reward_64[i], 4),
                rows$predations[i],
                fmt(rows$target_distance[i], 3),
                fmt(rows$brain_value_v[i], 4)))
  }
}
cat("  TOP:\n");    report_rows(head(last_rows, show_n))
if (nrow(last_rows) > show_n) { cat("  BOTTOM:\n"); report_rows(tail(last_rows, show_n)) }

cat(sprintf("\n  Population total predations (final per-organism sum): %d\n",
            sum(last_rows$predations, na.rm = TRUE)))

cat("\nDone.\n")
