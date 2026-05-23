#!/usr/bin/env Rscript
# general.R — routine population / behaviour / reward-channel diagnostics
#
# Reads every available `simulation_dataset_<milestone>.csv` from
# ../datasets/, builds a per-milestone snapshot of the herbivore
# cohort (with selected photoautotroph reference values), and writes
# a comprehensive report to ./results_general.txt.
#
# Usage:
#   cd data-analysis && Rscript general.R
#
# Or just:
#   Rscript data-analysis/general.R
#
# Either form works — the script resolves its own location and the
# `../datasets` path relative to it.

# ── Paths & milestone discovery ────────────────────────────────────

resolve_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit  <- args[grepl("^--file=", args)]
  if (length(hit) > 0) return(dirname(normalizePath(sub("^--file=", "", hit[1]))))
  # Fallback: assume current working directory.
  normalizePath(".")
}

script_dir   <- resolve_script_dir()
datasets_dir <- normalizePath(file.path(script_dir, "..", "datasets"), mustWork = FALSE)
output_path  <- file.path(script_dir, "results_general.txt")

# Milestones declared in the order they fire in the simulation. The
# script silently skips any that aren't present.
MILESTONES <- list(
  list(label = "5_MINUTES",  secs =   300),
  list(label = "10_MINUTES", secs =   600),
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

read_semis <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(
    read.csv(path, sep = ";", stringsAsFactors = FALSE, na.strings = c("", "NA")),
    error = function(e) { message(sprintf("  skip %s: %s", basename(path), conditionMessage(e))); NULL }
  )
}

available <- list()
for (m in MILESTONES) {
  p <- file.path(datasets_dir, sprintf("simulation_dataset_%s.csv", m$label))
  df <- read_semis(p)
  if (!is.null(df) && nrow(df) > 0) {
    available[[m$label]] <- list(label = m$label, secs = m$secs, df = df)
  }
}

if (length(available) == 0) {
  message(sprintf("No milestone CSVs found in %s", datasets_dir))
  quit(status = 1)
}

# ── Output sink ────────────────────────────────────────────────────

sink_file <- file(output_path, open = "w")
sink(sink_file, type = "output")
on.exit({ sink(type = "output"); close(sink_file) }, add = TRUE)

hdr <- function(txt, char = "═") {
  cat("\n", strrep(char, 78), "\n", sep = "")
  cat(" ", txt, "\n", sep = "")
  cat(strrep(char, 78), "\n", sep = "")
}
sub <- function(txt) {
  cat("\n── ", txt, " ", strrep("─", max(1, 70 - nchar(txt))), "\n", sep = "")
}
kv  <- function(k, v, w = 36) cat(sprintf("  %-*s %s\n", w, k, v))

# ── Header ─────────────────────────────────────────────────────────

cat("AEONS — General Run Report\n")
cat(sprintf("Generated at %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(sprintf("Datasets dir: %s\n", datasets_dir))
cat(sprintf("Milestones found: %d (%s)\n",
            length(available),
            paste(vapply(available, function(m) m$label, character(1)),
                  collapse = ", ")))

# Per-milestone safe-summary helpers.
fmt_num <- function(x, digits = 3) {
  if (is.null(x) || length(x) == 0) return("NA")
  if (is.numeric(x)) formatC(x, digits = digits, format = "f")
  else as.character(x)
}
safe_mean <- function(x) if (length(x) == 0 || all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
safe_med  <- function(x) if (length(x) == 0 || all(is.na(x))) NA_real_ else median(x, na.rm = TRUE)
safe_sd   <- function(x) if (length(x) == 0 || all(is.na(x))) NA_real_ else sd(x, na.rm = TRUE)
safe_min  <- function(x) if (length(x) == 0 || all(is.na(x))) NA_real_ else min(x, na.rm = TRUE)
safe_max  <- function(x) if (length(x) == 0 || all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
safe_q    <- function(x, q) if (length(x) == 0 || all(is.na(x))) NA_real_ else as.numeric(quantile(x, q, na.rm = TRUE))

describe_num <- function(x, label) {
  n_ok <- sum(!is.na(x))
  cat(sprintf("    %-26s n=%-5d  mean=%-9s med=%-9s sd=%-9s min=%-9s max=%-9s p10=%-9s p90=%s\n",
              label, n_ok,
              fmt_num(safe_mean(x)), fmt_num(safe_med(x)), fmt_num(safe_sd(x)),
              fmt_num(safe_min(x)),  fmt_num(safe_max(x)),
              fmt_num(safe_q(x, 0.10)), fmt_num(safe_q(x, 0.90))))
}

# ── 1) Per-milestone population snapshot ──────────────────────────

hdr("1) PER-MILESTONE SNAPSHOTS")

per_milestone_summary <- function(m) {
  df <- m$df
  sub(sprintf("Milestone %s  (virtual t = %d s, %d rows)", m$label, m$secs, nrow(df)))

  if ("trophic_class" %in% names(df)) {
    cls <- table(df$trophic_class)
    cat("  Trophic class:\n")
    for (n in names(cls)) kv(sprintf("  %s", n), as.integer(cls[[n]]))
  }

  if ("intelligence_level" %in% names(df)) {
    il <- table(df$intelligence_level)
    cat("  Intelligence level:\n")
    for (n in names(il)) kv(sprintf("  %s", n), as.integer(il[[n]]))
  }
  if ("symmetry" %in% names(df)) {
    sym <- table(df$symmetry)
    cat("  Symmetry:\n")
    for (n in names(sym)) kv(sprintf("  %s", n), as.integer(sym[[n]]))
  }
  if ("has_variable_form" %in% names(df)) {
    vf <- table(df$has_variable_form)
    cat("  Variable-form:\n")
    for (n in names(vf)) kv(sprintf("  %s", n), as.integer(vf[[n]]))
  }

  is_herb <- df$trophic_class == "Herbivore"
  is_carn <- df$trophic_class == "Carnivore"
  is_phot <- df$trophic_class == "Photoautotroph"
  herb    <- df[is_herb, ]
  carn    <- df[is_carn, ]
  phot    <- df[is_phot, ]

  cat(sprintf("\n  Herbivore subgroup (n=%d):\n", nrow(herb)))
  if (nrow(herb) > 0) {
    describe_num(herb$energy,         "energy")
    describe_num(herb$hunger,         "hunger")
    describe_num(herb$dopamine,       "dopamine")
    describe_num(herb$target_distance,"target_distance")
    describe_num(herb$movement_speed, "movement_speed")
    describe_num(herb$predations,     "predations (cumulative)")
    describe_num(herb$reproductions,  "reproductions")
    if ("age_secs" %in% names(herb))             describe_num(herb$age_secs,             "age_secs")
    if ("times_reproduced_self" %in% names(herb))describe_num(herb$times_reproduced_self,"times_reproduced_self")
    if ("child_alive_count" %in% names(herb))    describe_num(herb$child_alive_count,    "child_alive_count")
    if ("brain_mu_speed" %in% names(herb))       describe_num(herb$brain_mu_speed,       "brain_mu_speed")
    if ("brain_mu_angle" %in% names(herb))       describe_num(herb$brain_mu_angle,       "brain_mu_angle")
    if ("brain_log_sigma_speed" %in% names(herb))describe_num(herb$brain_log_sigma_speed,"brain_log_sigma_speed")
    if ("brain_log_sigma_angle" %in% names(herb))describe_num(herb$brain_log_sigma_angle,"brain_log_sigma_angle")
    if ("brain_value_v" %in% names(herb))        describe_num(herb$brain_value_v,        "brain_value_v")
    if ("brain_last_reward" %in% names(herb))    describe_num(herb$brain_last_reward,    "brain_last_reward")
    if ("brain_mean_reward_64" %in% names(herb)) describe_num(herb$brain_mean_reward_64, "brain_mean_reward_64")
    if ("brain_last_eat_component" %in% names(herb))      describe_num(herb$brain_last_eat_component,      "brain_last_eat_component")
    if ("brain_last_progress_component" %in% names(herb)) describe_num(herb$brain_last_progress_component, "brain_last_progress_component")
    if ("brain_last_oracle_component" %in% names(herb))   describe_num(herb$brain_last_oracle_component,   "brain_last_oracle_component")
  }

  if (nrow(carn) > 0) {
    cat(sprintf("\n  Carnivore subgroup (n=%d):\n", nrow(carn)))
    describe_num(carn$energy,     "energy")
    describe_num(carn$predations, "predations (cumulative)")
  }

  if (nrow(phot) > 0) {
    cat(sprintf("\n  Photoautotroph subgroup (n=%d):\n", nrow(phot)))
    describe_num(phot$energy,                "energy")
    describe_num(phot$grown_cell_count,      "grown_cell_count")
    describe_num(phot$alive_body_part_count, "alive_body_part_count")
    if ("age_secs" %in% names(phot)) describe_num(phot$age_secs, "age_secs")
  }

  # Reward-channel breakdown — restricted to herbivores with active
  # brains (non-zero brain_mu_speed or recent reward). Reports the
  # fraction of agents currently receiving non-zero signal on each
  # reward channel.
  if (nrow(herb) > 0 && "brain_last_oracle_component" %in% names(herb)) {
    cat("\n  Reward channel activity (herbivores):\n")
    frac_nonzero <- function(x) {
      n_ok <- sum(!is.na(x)); if (n_ok == 0) return(NA_real_)
      sum(!is.na(x) & x != 0) / n_ok
    }
    kv("  frac eat non-zero",      sprintf("%.1f%%",
       100 * frac_nonzero(herb$brain_last_eat_component)))
    kv("  frac progress non-zero", sprintf("%.1f%%",
       100 * frac_nonzero(herb$brain_last_progress_component)))
    kv("  frac oracle non-zero",   sprintf("%.1f%%",
       100 * frac_nonzero(herb$brain_last_oracle_component)))
    pos_frac <- function(x) {
      n_ok <- sum(!is.na(x)); if (n_ok == 0) return(NA_real_)
      sum(!is.na(x) & x > 0) / n_ok
    }
    kv("  frac oracle > 0 (toward prey)",   sprintf("%.1f%%",
       100 * pos_frac(herb$brain_last_oracle_component)))
    kv("  frac oracle < 0 (away from prey)",sprintf("%.1f%%",
       100 * (1 - pos_frac(herb$brain_last_oracle_component) -
              ifelse(is.na(frac_nonzero(herb$brain_last_oracle_component)),
                     NA_real_,
                     1 - frac_nonzero(herb$brain_last_oracle_component)))))
  }

  invisible(NULL)
}

for (m in available) per_milestone_summary(m)

# ── 2) Cross-milestone trends ──────────────────────────────────────

hdr("2) CROSS-MILESTONE TRENDS")

if (length(available) >= 2) {
  herb_mean <- function(df, col) {
    if (!(col %in% names(df))) return(NA_real_)
    x <- df[[col]][df$trophic_class == "Herbivore"]
    safe_mean(x)
  }
  herb_count <- function(df) sum(df$trophic_class == "Herbivore", na.rm = TRUE)
  phot_count <- function(df) sum(df$trophic_class == "Photoautotroph", na.rm = TRUE)

  cols_to_track <- c(
    "energy", "hunger", "dopamine", "movement_speed",
    "predations", "reproductions", "age_secs",
    "brain_mu_speed", "brain_mu_angle",
    "brain_log_sigma_speed", "brain_log_sigma_angle",
    "brain_value_v", "brain_last_reward", "brain_mean_reward_64",
    "brain_last_eat_component", "brain_last_progress_component",
    "brain_last_oracle_component"
  )

  # Wide table: rows = metric, cols = milestone.
  labels <- vapply(available, function(m) m$label, character(1))
  cat("Mean values of herbivore metrics across milestones\n")
  cat("(N/A means the column is missing in that milestone's CSV)\n\n")
  cat(sprintf("  %-32s", "metric"))
  for (lab in labels) cat(sprintf("  %12s", lab))
  cat("\n")
  cat(sprintf("  %-32s", "n_herbivores"))
  for (m in available) cat(sprintf("  %12d", herb_count(m$df)))
  cat("\n")
  cat(sprintf("  %-32s", "n_photoautotrophs"))
  for (m in available) cat(sprintf("  %12d", phot_count(m$df)))
  cat("\n")
  for (col in cols_to_track) {
    cat(sprintf("  %-32s", col))
    for (m in available) {
      v <- herb_mean(m$df, col)
      cat(sprintf("  %12s", if (is.na(v)) "N/A" else fmt_num(v, 4)))
    }
    cat("\n")
  }
}

# ── 3) Hunting / chase performance ─────────────────────────────────

hdr("3) HUNTING / CHASE PERFORMANCE (herbivores)")

if (length(available) >= 2) {
  # Predations per agent in each milestone, plus the delta between
  # adjacent milestones (rate per virtual second over the gap).
  cat("Per-milestone predation totals + per-agent and per-virtual-second rates\n\n")
  cat(sprintf("  %-12s  %10s  %14s  %18s  %16s\n",
              "milestone", "n_herb",
              "total_pred", "mean_pred / agent",
              "Δrate / agent / s"))
  prev <- NULL
  for (m in available) {
    herb <- m$df[m$df$trophic_class == "Herbivore", ]
    n_h  <- nrow(herb)
    tot  <- if (n_h > 0) sum(herb$predations, na.rm = TRUE) else 0
    mean_pa <- if (n_h > 0) safe_mean(herb$predations) else NA_real_

    rate <- if (!is.null(prev)) {
      gap_s   <- m$secs - prev$secs
      d_total <- tot - prev$total
      # Per-agent rate: use the average of the two cohorts
      avg_n   <- (n_h + prev$n) / 2
      if (gap_s > 0 && avg_n > 0) d_total / (gap_s * avg_n) else NA_real_
    } else NA_real_

    cat(sprintf("  %-12s  %10d  %14d  %18s  %16s\n",
                m$label, n_h, tot,
                fmt_num(mean_pa, 3),
                if (is.na(rate)) "—" else fmt_num(rate, 5)))
    prev <- list(total = tot, n = n_h, secs = m$secs)
  }
  cat("\n")
  cat("Interpretation:\n")
  cat("  * Rising mean_pred/agent indicates herbivores accumulate predations.\n")
  cat("  * Rising Δrate/agent/s indicates each herbivore is eating MORE\n")
  cat("    PER SECOND than in the previous milestone — i.e. the policy is\n")
  cat("    learning to chase. A flat or falling rate means random-walk-style\n")
  cat("    encounters.\n")
}

# ── 4) Brain-policy alignment with oracle direction ────────────────

hdr("4) POLICY-MEAN ALIGNMENT WITH ORACLE")

# `brain_mu_angle` is the pre-tanh output. After tanh it's in [-1, 1]
# and PI*tanh(.) is the body-frame turn. The expected sign of the
# optimal turn matches the sign of body_right in the observation.
# We don't have body_right in this CSV directly, but we do have the
# oracle reward component; if the policy was previously learning to
# chase, mu_angle's sign should correlate with the sign of the
# oracle reward (which is +ve when moving toward prey).

for (m in available) {
  herb <- m$df[m$df$trophic_class == "Herbivore", ]
  if (nrow(herb) < 5) next
  if (!all(c("brain_mu_angle", "brain_last_oracle_component") %in% names(herb))) next

  sub(sprintf("%s — alignment μ_angle ↔ oracle reward", m$label))

  mu  <- herb$brain_mu_angle
  ora <- herb$brain_last_oracle_component
  ok  <- !is.na(mu) & !is.na(ora)
  if (sum(ok) < 5) {
    cat("  insufficient non-NA pairs\n")
    next
  }
  var_mu <- var(mu[ok])
  var_or <- var(ora[ok])
  rho_p  <- if (var_mu > 1e-12 && var_or > 1e-12)
              suppressWarnings(cor(mu[ok], ora[ok], method = "pearson")) else NA_real_
  rho_s  <- if (var_mu > 1e-12 && var_or > 1e-12)
              suppressWarnings(cor(mu[ok], ora[ok], method = "spearman")) else NA_real_

  kv("n",                 sum(ok))
  kv("var(μ_angle)",      fmt_num(var_mu, 6))
  kv("var(oracle reward)",fmt_num(var_or, 6))
  kv("Pearson ρ",         fmt_num(rho_p, 3))
  kv("Spearman ρ",        fmt_num(rho_s, 3))
  if (is.na(rho_p)) {
    cat("  Note: zero variance — policy hasn't differentiated yet.\n")
  }
}

# ── 5) Reproduction & lineage health ──────────────────────────────

hdr("5) REPRODUCTION & LINEAGE")

for (m in available) {
  sub(sprintf("%s lineage", m$label))
  df <- m$df
  if ("parent_id" %in% names(df)) {
    has_parent <- !is.na(df$parent_id) & df$parent_id != ""
    n_with     <- sum(has_parent)
    n_without  <- sum(!has_parent)
    kv("Organisms with parent_id (offspring)", n_with)
    kv("Organisms without (initial/cohort)",   n_without)
  }
  if ("times_reproduced_self" %in% names(df)) {
    trs <- df$times_reproduced_self
    kv("max times_reproduced_self", safe_max(trs))
    kv("mean", fmt_num(safe_mean(trs)))
    kv("frac with ≥ 1 reproduction",
       sprintf("%.1f%%", 100 * (sum(trs > 0, na.rm = TRUE) / max(1, length(trs[!is.na(trs)])))))
  }
  if ("child_alive_count" %in% names(df)) {
    cac <- df$child_alive_count
    kv("max child_alive_count", safe_max(cac))
    kv("mean", fmt_num(safe_mean(cac)))
  }
  if ("species_id" %in% names(df)) {
    species <- unique(df$species_id[!is.na(df$species_id)])
    kv("distinct species_id present", length(species))
  }
}

# ── 6) Δ predations across adjacent milestones (matched cohort) ─

hdr("6) Δ PREDATIONS / Δ REPRODUCTIONS (matched cohort)")

if (length(available) >= 2) {
  ms <- available
  for (i in 2:length(ms)) {
    a <- ms[[i - 1]]; b <- ms[[i]]
    if (!all(c("entity_id", "predations") %in% names(a$df))) next
    sub(sprintf("%s → %s  (gap = %d s)", a$label, b$label, b$secs - a$secs))

    ah <- a$df[a$df$trophic_class == "Herbivore", c("entity_id", "predations", "reproductions")]
    bh <- b$df[b$df$trophic_class == "Herbivore", c("entity_id", "predations", "reproductions")]
    j  <- merge(ah, bh, by = "entity_id", suffixes = c(".a", ".b"))
    if (nrow(j) == 0) {
      cat("  no entity_id overlap (no individual-level continuity between runs)\n")
      next
    }
    dpred  <- j$predations.b    - j$predations.a
    drepro <- j$reproductions.b - j$reproductions.a
    kv("matched herbivores", nrow(j))
    kv("Δpredations mean",    fmt_num(safe_mean(dpred), 3))
    kv("Δpredations median",  fmt_num(safe_med(dpred), 3))
    kv("Δpredations max",     fmt_num(safe_max(dpred), 3))
    kv("frac with Δpred > 0", sprintf("%.1f%%",
       100 * sum(dpred > 0, na.rm = TRUE) / max(1, nrow(j))))
    kv("Δreproductions mean", fmt_num(safe_mean(drepro), 3))
    kv("Δreproductions max",  fmt_num(safe_max(drepro), 3))
  }
}

# ── 7) DNA diversity ──────────────────────────────────────────────

hdr("7) DNA DIVERSITY (latest milestone)")

latest <- available[[length(available)]]
df_l <- latest$df
dna_cols <- grep("^dna_", names(df_l), value = TRUE)
if (length(dna_cols) > 0) {
  cat(sprintf("(Computed on %s, %d herbivores)\n\n",
              latest$label,
              sum(df_l$trophic_class == "Herbivore")))
  herb_l <- df_l[df_l$trophic_class == "Herbivore", ]
  if (nrow(herb_l) > 0) {
    for (c in dna_cols) {
      x <- herb_l[[c]]
      describe_num(x, c)
    }
  }
}

cat("\n=== END ===\n")
sink(type = "output")
close(sink_file)
on.exit()  # clear

message(sprintf("wrote %s", output_path))
