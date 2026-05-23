#!/usr/bin/env Rscript
# training_curves.R — loss / return / σ / supervised-loss diagnostics
#
# Reads every available `simulation_dataset_<milestone>_training_stats.csv`
# from ../datasets/, analyses each independently, and compares the
# earliest vs the latest milestone. Writes a comprehensive report
# to ./results_training_curves.txt covering:
#
#   1. Per-milestone snapshot summaries
#   2. Trend-test (linear regression on step) for every metric
#   3. Loss decomposition (actor / critic / -ENTROPY_COEF·entropy /
#      supervised) and trend per channel
#   4. Critic sanity check (ρ between critic_loss and return_var)
#   5. Supervised bootstrap decay verification
#   6. Change-point detection on mean_return + losses
#   7. Cross-milestone endpoint comparison
#
# Usage:
#   Rscript data-analysis/training_curves.R

resolve_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit  <- args[grepl("^--file=", args)]
  if (length(hit) > 0) return(dirname(normalizePath(sub("^--file=", "", hit[1]))))
  normalizePath(".")
}
script_dir   <- resolve_script_dir()
datasets_dir <- normalizePath(file.path(script_dir, "..", "datasets"), mustWork = FALSE)
output_path  <- file.path(script_dir, "results_training_curves.txt")

# These constants mirror `intelligence_level_herbivore_1.rs` and are
# used by the loss-decomposition step. Kept in sync manually — if
# they ever change in the Rust source, update here.
VALUE_COEF   <- 0.5
ENTROPY_COEF <- 0.0   # zero since the May-2026 fix
BOOTSTRAP_STEPS <- 200

MILESTONES <- c("5_MINUTES","10_MINUTES","30_MINUTES","60_MINUTES",
                "3_HOURS","6_HOURS","12_HOURS","24_HOURS",
                "48_HOURS","96_HOURS","156_HOURS")
MILESTONE_SECS <- c(300, 600, 1800, 3600,
                    10800, 21600, 43200, 86400,
                    172800, 345600, 561600)

read_semis <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(
    read.csv(path, sep = ";", stringsAsFactors = FALSE, na.strings = c("", "NA")),
    error = function(e) NULL
  )
}

stats_list <- list()
for (i in seq_along(MILESTONES)) {
  lab  <- MILESTONES[i]
  secs <- MILESTONE_SECS[i]
  p    <- read_semis(file.path(datasets_dir,
                               sprintf("simulation_dataset_%s_training_stats.csv", lab)))
  if (!is.null(p) && nrow(p) > 0) {
    stats_list[[lab]] <- list(label = lab, secs = secs, df = p)
  }
}

if (length(stats_list) == 0) {
  message("No training_stats CSVs found.")
  quit(status = 1)
}

sink_file <- file(output_path, open = "w")
sink(sink_file, type = "output")
on.exit({ sink(type = "output"); close(sink_file) }, add = TRUE)

hdr <- function(t, ch = "═") {
  cat("\n", strrep(ch, 78), "\n", sep = "")
  cat(" ", t, "\n", sep = "")
  cat(strrep(ch, 78), "\n", sep = "")
}
sub <- function(t) cat("\n── ", t, " ", strrep("─", max(1, 70 - nchar(t))), "\n", sep = "")
kv  <- function(k, v, w = 36) cat(sprintf("  %-*s %s\n", w, k, v))
fmt <- function(x, d = 4) {
  if (is.null(x) || length(x) == 0 || !is.finite(x)) "NA"
  else formatC(x, digits = d, format = "f")
}
fmtg <- function(x, d = 3) {
  if (is.null(x) || length(x) == 0 || !is.finite(x)) "NA"
  else formatC(x, digits = d, format = "g")
}
safe_mean <- function(x) if (length(x) == 0 || all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_sd   <- function(x) if (length(x) == 0 || all(is.na(x))) NA_real_ else sd(x, na.rm = TRUE)

cat("AEONS — Training Curves Report\n")
cat(sprintf("Generated at %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(sprintf("Milestones found: %d (%s)\n",
            length(stats_list),
            paste(vapply(stats_list, function(p) p$label, character(1)), collapse = ", ")))
cat(sprintf("Reference constants: VALUE_COEF=%.2f, ENTROPY_COEF=%.3f, BOOTSTRAP_STEPS=%d\n",
            VALUE_COEF, ENTROPY_COEF, BOOTSTRAP_STEPS))

# Helper to summarise one numeric column over the rollout step axis.
describe <- function(x, name) {
  cat(sprintf("    %-20s n=%-5d  mean=%-11s sd=%-11s min=%-11s max=%-11s\n",
              name, sum(!is.na(x)),
              fmt(safe_mean(x), 5),
              fmt(safe_sd(x), 5),
              fmt(min(x, na.rm = TRUE), 5),
              fmt(max(x, na.rm = TRUE), 5)))
}

trend_test <- function(y, x) {
  ok <- is.finite(y) & is.finite(x)
  if (sum(ok) < 5) return(list(slope = NA_real_, p = NA_real_, R2 = NA_real_, n = sum(ok)))
  m  <- tryCatch(lm(y[ok] ~ x[ok]), error = function(e) NULL)
  if (is.null(m)) return(list(slope = NA_real_, p = NA_real_, R2 = NA_real_, n = sum(ok)))
  cf <- coef(summary(m))
  list(slope = cf[2, 1], p = cf[2, 4],
       R2 = summary(m)$r.squared, n = sum(ok))
}

single_change_point <- function(x) {
  ok <- is.finite(x)
  x  <- x[ok]
  n  <- length(x)
  if (n < 40) return(list(idx = NA_integer_, delta = NA_real_))
  best <- list(idx = NA_integer_, delta = 0)
  for (k in seq(20, n - 20)) {
    pre <- mean(x[1:k]); post <- mean(x[(k+1):n])
    d <- post - pre
    if (abs(d) > abs(best$delta)) best <- list(idx = k, delta = d)
  }
  best
}

rolling_mean <- function(x, w = 50) {
  n <- length(x); out <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    lo <- max(1, i - w + 1)
    out[i] <- mean(x[lo:i], na.rm = TRUE)
  }
  out
}

# ── 1) Per-milestone snapshot summaries ──────────────────────────

hdr("1) PER-MILESTONE SUMMARIES")

per_summary <- function(s) {
  df <- s$df
  sub(sprintf("%s  (virtual t = %d s, %d steps logged)",
              s$label, s$secs, nrow(df)))
  cat(sprintf("    virtual_time range: %.2f s → %.2f s   step range: %d → %d\n",
              min(df$virtual_time_secs), max(df$virtual_time_secs),
              min(df$step), max(df$step)))
  for (col in c("actor_loss", "critic_loss", "entropy", "total_loss",
                "mean_return", "return_var", "supervised_loss")) {
    if (col %in% names(df)) describe(df[[col]], col)
  }
  if ("n_active" %in% names(df)) describe(df$n_active, "n_active")
}

for (s in stats_list) per_summary(s)

# ── 2) Trend tests per metric per milestone ──────────────────────

hdr("2) TREND TESTS (linear regression on step)")

metrics <- c("actor_loss", "critic_loss", "entropy",
             "total_loss", "mean_return", "return_var",
             "supervised_loss")

cat(sprintf("\n  %-12s  %-16s  %12s  %12s  %8s  %6s\n",
            "milestone", "metric", "slope", "p", "R²", "n"))
for (s in stats_list) {
  for (col in metrics) {
    if (!(col %in% names(s$df))) next
    r <- trend_test(s$df[[col]], s$df$step)
    cat(sprintf("  %-12s  %-16s  %12s  %12s  %8s  %6d\n",
                s$label, col,
                fmtg(r$slope, 4), fmtg(r$p, 3),
                fmt (r$R2,    3), r$n))
  }
}

cat("\nInterpretation:\n")
cat("  * Negative slope + small p on actor_loss / critic_loss / total_loss\n")
cat("    means the optimizer is converging. The MAGNITUDE of the slope is\n")
cat("    important — slopes of 1e-6 or smaller mean the curve is essentially\n")
cat("    flat (statistically detectable but practically zero).\n")
cat("  * Positive slope on mean_return = the policy is collecting more\n")
cat("    reward over time (learning to chase). Negative or flat = stuck.\n")
cat("  * Negative slope on entropy is meaningless when ENTROPY_COEF = 0\n")
cat("    (entropy is not part of the loss any more) — included for sanity.\n")
cat("  * Supervised_loss should be > 0 only for the first BOOTSTRAP_STEPS\n")
cat("    training steps and exactly 0 after — see Step 5 below.\n")

# ── 3) Loss decomposition (latest milestone) ─────────────────────

hdr("3) LOSS DECOMPOSITION")

latest <- stats_list[[length(stats_list)]]
df_l   <- latest$df

cat(sprintf("\nDecomposition computed on %s (the latest milestone).\n", latest$label))
cat("Components:\n")
cat("  actor_contrib    = actor_loss\n")
cat(sprintf("  critic_contrib   = %.2f * critic_loss\n", VALUE_COEF))
cat(sprintf("  entropy_contrib  = -%.3f * entropy\n",    ENTROPY_COEF))
cat("  supervised_contr = supervised_loss (already scaled by bootstrap_weight in Rust)\n\n")

if (all(c("actor_loss","critic_loss","entropy","total_loss","supervised_loss") %in% names(df_l))) {
  df_l$actor_contrib   <- df_l$actor_loss
  df_l$critic_contrib  <- VALUE_COEF * df_l$critic_loss
  df_l$entropy_contrib <- -ENTROPY_COEF * df_l$entropy
  df_l$sup_contrib     <- df_l$supervised_loss
  df_l$reconstructed   <- df_l$actor_contrib + df_l$critic_contrib +
                          df_l$entropy_contrib + df_l$sup_contrib
  resid <- df_l$total_loss - df_l$reconstructed
  kv("Reconstruction residual mean", fmtg(safe_mean(resid), 4))
  kv("Reconstruction residual max|.|",fmtg(max(abs(resid), na.rm = TRUE), 4))
  if (max(abs(resid), na.rm = TRUE) > 1e-3) {
    cat("  Note: residual is non-trivial — there may be additional terms\n")
    cat("        (e.g. weight decay) in the Rust loss that aren't accounted\n")
    cat("        for here. Or VALUE_COEF / ENTROPY_COEF have drifted.\n")
  }

  cat("\nMean contribution to total_loss (absolute share):\n")
  components <- c("actor_contrib", "critic_contrib", "entropy_contrib", "sup_contrib")
  means <- sapply(components, function(c) safe_mean(df_l[[c]]))
  shares <- abs(means) / sum(abs(means))
  for (i in seq_along(components)) {
    kv(components[i],
       sprintf("mean=%-12s   share=%4.1f%%",
               fmt(means[i], 5), 100 * shares[i]))
  }

  cat("\nTrend slope per channel (slope per step):\n")
  cat(sprintf("  %-18s  %12s  %12s  %8s  %6s\n",
              "channel", "slope", "p", "R²", "n"))
  for (c in components) {
    r <- trend_test(df_l[[c]], df_l$step)
    cat(sprintf("  %-18s  %12s  %12s  %8s  %6d\n",
                c, fmtg(r$slope, 4), fmtg(r$p, 3),
                fmt(r$R2, 3), r$n))
  }
}

# ── 4) Critic sanity check ────────────────────────────────────────

hdr("4) CRITIC SANITY CHECK")

cat("Question: is critic_loss going down because the critic is learning,\n")
cat("or because the targets (returns) are themselves shrinking?\n")
cat("Spearman ρ(critic_loss, return_var) ≈ 1 → critic just tracks the\n")
cat("targets; no real prediction skill. Lower ρ → critic is independent.\n\n")

for (s in stats_list) {
  if (!all(c("critic_loss", "return_var", "mean_return") %in% names(s$df))) next
  r1 <- suppressWarnings(cor(s$df$critic_loss, s$df$return_var,        method = "spearman"))
  r2 <- suppressWarnings(cor(s$df$critic_loss, abs(s$df$mean_return),  method = "spearman"))
  cat(sprintf("  %-12s  ρ(critic_loss, return_var) = %+.3f   ρ(critic_loss, |mean_return|) = %+.3f\n",
              s$label, r1, r2))
}

# ── 5) Supervised bootstrap decay ─────────────────────────────────

hdr("5) SUPERVISED BOOTSTRAP DECAY")

cat(sprintf("The supervised pull is scaled by (1 - step/%d) for step < %d\n",
            BOOTSTRAP_STEPS, BOOTSTRAP_STEPS))
cat("and 0 thereafter. The CSV column `supervised_loss` is the contribution\n")
cat("AFTER that scaling, so it should drop to zero by step %d.\n\n",
    BOOTSTRAP_STEPS)

for (s in stats_list) {
  if (!("supervised_loss" %in% names(s$df))) next
  sub(sprintf("%s", s$label))
  sup <- s$df$supervised_loss
  step <- s$df$step
  # Three regions: step < 100 (early), 100..200 (decay tail),
  # > 200 (should be zero)
  e1 <- sup[step <  100]
  e2 <- sup[step >= 100 & step <= 200]
  e3 <- sup[step >  200]
  kv("steps < 100   — mean sup_loss", fmt(safe_mean(e1), 4))
  kv("steps 100..200 — mean sup_loss",fmt(safe_mean(e2), 4))
  kv("steps > 200    — mean sup_loss",fmt(safe_mean(e3), 6))
  kv("steps > 200    — max  sup_loss",fmt(max(e3, na.rm = TRUE), 6))
  if (length(e3) > 0 && safe_mean(e3) > 0.001) {
    cat("  ⚠ supervised_loss is non-zero past step 200 — bootstrap didn't\n")
    cat("    decay as expected. Inspect intelligence_level_herbivore_1.rs.\n")
  } else if (length(e3) > 0) {
    cat("  ✓ supervised channel cleanly zeroed out post-bootstrap.\n")
  }
}

# ── 6) Change-point detection ─────────────────────────────────────

hdr("6) CHANGE-POINT DETECTION (binary CUSUM)")

cat("Best single breakpoint for each metric within each milestone.\n")
cat("Useful to spot when learning starts (or stops).\n\n")

for (s in stats_list) {
  sub(sprintf("%s — %d steps", s$label, nrow(s$df)))
  for (col in c("mean_return", "actor_loss", "critic_loss", "total_loss")) {
    if (!(col %in% names(s$df))) next
    cp <- single_change_point(s$df[[col]])
    if (is.na(cp$idx)) {
      kv(col, "(too few steps)")
    } else {
      kv(col,
         sprintf("break at step %d (Δ post-pre = %+.4f)",
                 cp$idx, cp$delta))
    }
  }
}

# ── 7) Endpoint comparison ────────────────────────────────────────

hdr("7) ENDPOINT COMPARISON ACROSS MILESTONES")

cat("Last-50-step rolling mean of each metric, per milestone.\n")
cat("Compare adjacent rows to see how training has progressed.\n\n")
endpoints <- function(s) {
  df <- s$df; n <- nrow(df)
  if (n < 5) return(NULL)
  k <- min(50, n)
  tail_rows <- df[(n - k + 1):n, , drop = FALSE]
  vapply(c("actor_loss","critic_loss","entropy","total_loss",
           "mean_return","return_var","supervised_loss"),
         function(col) if (col %in% names(df)) safe_mean(tail_rows[[col]]) else NA_real_,
         numeric(1))
}

# Print as table.
metrics_e <- c("actor_loss","critic_loss","entropy","total_loss",
               "mean_return","return_var","supervised_loss")
cat(sprintf("  %-12s", "milestone"))
for (mt in metrics_e) cat(sprintf("  %14s", mt))
cat("\n")
for (s in stats_list) {
  cat(sprintf("  %-12s", s$label))
  e <- endpoints(s)
  for (mt in metrics_e) {
    cat(sprintf("  %14s", if (is.na(e[mt])) "—" else fmt(e[mt], 5)))
  }
  cat("\n")
}

# ── 8) Synthesis ──────────────────────────────────────────────────

hdr("8) SYNTHESIS")

# A few derived verdicts based on the latest milestone.
df_l <- latest$df
verdict <- function(label, pass, detail) {
  mark <- if (is.na(pass)) "N/A " else if (pass) "PASS" else "FAIL"
  cat(sprintf("  %s  %-44s  %s\n", mark, label, detail))
}

if ("total_loss" %in% names(df_l) && "step" %in% names(df_l)) {
  r <- trend_test(df_l$total_loss, df_l$step)
  verdict("Total loss trending down",
          !is.na(r$slope) && r$slope < 0 && !is.na(r$p) && r$p < 0.05,
          sprintf("slope=%s, p=%s",
                  fmtg(r$slope, 3), fmtg(r$p, 3)))
}
if ("mean_return" %in% names(df_l)) {
  r <- trend_test(df_l$mean_return, df_l$step)
  verdict("Mean return rising",
          !is.na(r$slope) && r$slope > 0 && !is.na(r$p) && r$p < 0.05,
          sprintf("slope=%s, p=%s",
                  fmtg(r$slope, 3), fmtg(r$p, 3)))
}
if (all(c("critic_loss","return_var") %in% names(df_l))) {
  rho <- suppressWarnings(cor(df_l$critic_loss, df_l$return_var, method = "spearman"))
  verdict("Critic decoupled from return_var (ρ < 0.95)",
          !is.na(rho) && rho < 0.95,
          sprintf("ρ=%s", fmt(rho, 3)))
}
if ("supervised_loss" %in% names(df_l)) {
  late_steps <- df_l$supervised_loss[df_l$step > BOOTSTRAP_STEPS]
  if (length(late_steps) > 0) {
    verdict("Supervised bootstrap zeroed post-decay",
            mean(late_steps, na.rm = TRUE) < 0.001,
            sprintf("post-step-%d mean = %s",
                    BOOTSTRAP_STEPS, fmt(mean(late_steps, na.rm = TRUE), 5)))
  } else {
    verdict("Supervised bootstrap zeroed post-decay", NA,
            "(not enough steps past bootstrap)")
  }
}

cat("\n=== END ===\n")
sink(type = "output")
close(sink_file)
on.exit()
message(sprintf("wrote %s", output_path))
