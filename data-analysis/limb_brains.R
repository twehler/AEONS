#!/usr/bin/env Rscript
# limb_brains.R — diagnostics for the limb-based PPO pools
#
# Reads every available limb-pool side-car CSV from `../datasets/`:
#
#   simulation_dataset_<MILESTONE>_limb_herbivore_1.csv
#   simulation_dataset_<MILESTONE>_limb_l2.csv
#   simulation_dataset_<MILESTONE>_limb_l3.csv
#
# These are written by `dataset_export.rs::write_limb_pool_csv` next to
# the main `simulation_dataset_<MILESTONE>.csv`. Each row is one
# enrolled organism's current state + actor `log_std` per action dim.
#
# Output: ./results_limb_brains.txt — population counts per pool per
# milestone, exploration (mean `exp(log_std)`) trends over time, plus
# per-pool current-action and energy summaries.
#
# Usage:
#   cd data-analysis && Rscript limb_brains.R
# or
#   Rscript data-analysis/limb_brains.R

# ── Paths & milestone discovery ────────────────────────────────────

resolve_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit  <- args[grepl("^--file=", args)]
  if (length(hit) > 0) return(dirname(normalizePath(sub("^--file=", "", hit[1]))))
  normalizePath(".")
}

script_dir   <- resolve_script_dir()
datasets_dir <- normalizePath(file.path(script_dir, "..", "datasets"), mustWork = FALSE)
output_path  <- file.path(script_dir, "results_limb_brains.txt")

# Mirrored from general.R so the milestone set stays consistent.
MILESTONES <- list(
  list(label = "0_SECONDS",  secs =     0),
  list(label = "1_MINUTE",   secs =    60),
  list(label = "3_MINUTES",  secs =   180),
  list(label = "5_MINUTES",  secs =   300),
  list(label = "7_MINUTES",  secs =   420),
  list(label = "10_MINUTES", secs =   600),
  list(label = "20_MINUTES", secs =  1200),
  list(label = "30_MINUTES", secs =  1800),
  list(label = "60_MINUTES", secs =  3600),
  list(label = "3_HOURS",    secs = 10800),
  list(label = "6_HOURS",    secs = 21600),
  list(label = "12_HOURS",   secs = 43200),
  list(label = "24_HOURS",   secs = 86400),
  list(label = "48_HOURS",   secs =172800),
  list(label = "96_HOURS",   secs =345600),
  list(label = "156_HOURS",  secs =561600)
)

POOLS <- c("limb_herbivore_1", "limb_l2", "limb_l3")

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

# Discover (milestone, pool) → data-frame map.
records <- list()
for (m in MILESTONES) {
  for (p in POOLS) {
    path <- file.path(
      datasets_dir,
      sprintf("simulation_dataset_%s_%s.csv", m$label, p)
    )
    df <- read_semis(path)
    if (!is.null(df) && nrow(df) > 0) {
      records[[length(records) + 1]] <- list(
        milestone = m$label,
        secs      = m$secs,
        pool      = p,
        df        = df
      )
    }
  }
}

if (length(records) == 0) {
  message(sprintf("No limb-pool CSVs found in %s", datasets_dir))
  message("Run the simulation with `Export Simulation Dataset` (or wait for an autosave milestone) ",
          "and confirm limb-based organisms were alive at the time.")
  quit(status = 1)
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

# ── Per-(milestone × pool) summary block ───────────────────────────

hdr("Limb-Brain Diagnostics", "=")
cat("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\n", sep = "")
cat("Datasets dir: ", datasets_dir, "\n", sep = "")
cat("Records found: ", length(records), "\n", sep = "")

for (r in records) {
  hdr(sprintf("%s — %s (%d organisms)", r$milestone, r$pool, nrow(r$df)), "-")

  df <- r$df

  # Population: how many organisms, energy summary, action energy.
  cat(sprintf("  Population: %d enrolled\n", nrow(df)))
  cat(sprintf("  Mean energy_norm:    %s\n", fmt_num(mean(df$energy_norm,    na.rm = TRUE))))
  cat(sprintf("  Mean predations:     %s\n", fmt_num(mean(df$predations,     na.rm = TRUE))))
  cat(sprintf("  Mean reproductions:  %s\n", fmt_num(mean(df$reproductions,  na.rm = TRUE))))

  # Per-action-dim log_std summary — exploration breadth on each DOF.
  cat("  log_std per action dim (mean / sd):\n")
  for (d in 0:5) {
    col <- sprintf("log_std_%d", d)
    if (!is.null(df[[col]])) {
      mu <- mean(df[[col]], na.rm = TRUE)
      sd <- sd(df[[col]],   na.rm = TRUE)
      cat(sprintf("    dim %d: mean=%s sd=%s  (sigma ≈ %s)\n",
                  d, fmt_num(mu), fmt_num(sd), fmt_num(exp(mu))))
    }
  }

  # Limb-target activity per DOF (current commanded angle).
  cat("  limb_target magnitude per dim (mean abs):\n")
  for (d in 0:5) {
    col <- sprintf("limb_target_%d", d)
    if (!is.null(df[[col]])) {
      cat(sprintf("    dim %d: %s\n", d, fmt_num(mean(abs(df[[col]]), na.rm = TRUE))))
    }
  }
}

# ── Cross-milestone trends ─────────────────────────────────────────
# For each pool, plot exploration σ (= mean exp(log_std)) over time
# to see whether PPO is collapsing variance (exploitation taking over)
# or holding open exploration.

hdr("Cross-Milestone Trends", "=")

for (p in POOLS) {
  rec_for_pool <- Filter(function(r) r$pool == p, records)
  if (length(rec_for_pool) == 0) next

  cat(sprintf("\n[%s] Exploration σ + reward proxies by milestone:\n", p))
  cat("  milestone           secs     N    mean_sigma    mean_energy   mean_predations\n")
  cat("  ----------------  -------   ---  ------------  ------------  ----------------\n")

  for (r in rec_for_pool) {
    df <- r$df
    # Mean σ across all log_std columns, then exp.
    log_std_cols <- grep("^log_std_", names(df), value = TRUE)
    sigmas <- if (length(log_std_cols) > 0) {
      means_per_dim <- sapply(log_std_cols, function(c) mean(df[[c]], na.rm = TRUE))
      mean(exp(means_per_dim))
    } else { NA_real_ }
    cat(sprintf("  %-16s  %7d  %4d  %12s  %12s  %16s\n",
                r$milestone, r$secs, nrow(df),
                fmt_num(sigmas, 4),
                fmt_num(mean(df$energy_norm, na.rm = TRUE), 4),
                fmt_num(mean(df$predations, na.rm = TRUE), 2)))
  }
}

cat("\nDone.\n")
