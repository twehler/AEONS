#!/usr/bin/env Rscript
# brain_weights.R — per-tensor evolution + per-organism fitness link
#
# Reads every available `simulation_dataset_<milestone>_brain_probe.csv`
# from ../datasets/ alongside the matching colony CSV (for the
# predations column). Writes a comprehensive report to
# ./results_brain_weights.txt covering:
#
#   1. Per-milestone per-tensor distribution (l2_norm, std, |mean|)
#   2. Cross-milestone shift (mean change, KS test for distribution
#      shift, Welch t-test for mean change)
#   3. Bias-decay diagnostic (a_b2 / c_b2 / etc.)
#   4. Output-layer magnitude bound + napkin estimate of μ saturation
#   5. Within-milestone weight-vs-predations correlation per tensor
#   6. Top-mover vs bottom-mover comparison on actor / critic outputs
#
# Usage:
#   Rscript data-analysis/brain_weights.R

resolve_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit  <- args[grepl("^--file=", args)]
  if (length(hit) > 0) return(dirname(normalizePath(sub("^--file=", "", hit[1]))))
  normalizePath(".")
}
script_dir   <- resolve_script_dir()
datasets_dir <- normalizePath(file.path(script_dir, "..", "datasets"), mustWork = FALSE)
output_path  <- file.path(script_dir, "results_brain_weights.txt")

MILESTONES <- c("0_SECONDS",
                "1_MINUTE","3_MINUTES","5_MINUTES","7_MINUTES",
                "10_MINUTES","20_MINUTES","30_MINUTES","60_MINUTES",
                "3_HOURS","6_HOURS","12_HOURS","24_HOURS",
                "48_HOURS","96_HOURS","156_HOURS")
MILESTONE_SECS <- c(0,
                    60, 180, 300, 420,
                    600, 1200, 1800, 3600,
                    10800, 21600, 43200, 86400,
                    172800, 345600, 561600)

read_semis <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch(
    read.csv(path, sep = ";", stringsAsFactors = FALSE, na.strings = c("", "NA")),
    error = function(e) NULL
  )
}

probes  <- list()
colony  <- list()
for (i in seq_along(MILESTONES)) {
  lab <- MILESTONES[i]; secs <- MILESTONE_SECS[i]
  p <- read_semis(file.path(datasets_dir, sprintf("simulation_dataset_%s_brain_probe.csv", lab)))
  c <- read_semis(file.path(datasets_dir, sprintf("simulation_dataset_%s.csv",             lab)))
  if (!is.null(p) && nrow(p) > 0) {
    probes[[lab]]  <- list(label = lab, secs = secs, df = p)
    colony[[lab]]  <- c
  }
}

if (length(probes) == 0) {
  message("No brain_probe CSVs found.")
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

cat("AEONS — Brain Weights Report\n")
cat(sprintf("Generated at %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(sprintf("Milestones with brain_probe: %d (%s)\n",
            length(probes),
            paste(vapply(probes, function(p) p$label, character(1)), collapse = ", ")))

# Discover the tensor names from the first available probe.
tensors <- unique(probes[[1]]$df$tensor)
cat(sprintf("Tensors in probe: %s\n", paste(tensors, collapse = ", ")))

# ── 1) Per-milestone per-tensor distribution ──────────────────────

hdr("1) PER-MILESTONE PER-TENSOR DISTRIBUTION")

# Wide table: rows = tensor, cols = milestone. Stats: mean(l2), std(l2).
labels <- vapply(probes, function(p) p$label, character(1))
print_metric_table <- function(metric_fn, metric_name) {
  cat(sprintf("\n%s\n", metric_name))
  cat(sprintf("  %-10s", "tensor"))
  for (l in labels) cat(sprintf("  %12s", l))
  cat("\n")
  for (tn in tensors) {
    cat(sprintf("  %-10s", tn))
    for (p in probes) {
      sub_df <- p$df[p$df$tensor == tn, ]
      cat(sprintf("  %12s", fmt(metric_fn(sub_df), 4)))
    }
    cat("\n")
  }
}
print_metric_table(function(d) mean(d$l2_norm,  na.rm = TRUE), "Mean of l2_norm per tensor (per-organism)")
print_metric_table(function(d) sd  (d$l2_norm,  na.rm = TRUE), "Std  of l2_norm per tensor (per-organism)")
print_metric_table(function(d) mean(d$std,      na.rm = TRUE), "Mean of element-std per tensor")
print_metric_table(function(d) mean(abs(d$mean), na.rm = TRUE),"Mean of |mean| per tensor")

# ── 2) Cross-milestone shift (first vs last) ──────────────────────

hdr("2) DISTRIBUTION SHIFT: FIRST vs LAST MILESTONE")

if (length(probes) >= 2) {
  first <- probes[[1]]
  last  <- probes[[length(probes)]]
  cat(sprintf("First milestone: %s   Last milestone: %s\n\n",
              first$label, last$label))

  cat(sprintf("  %-10s  %12s  %12s  %12s  %10s  %12s  %10s\n",
              "tensor", paste0("mean_", first$label),
              paste0("mean_", last$label),
              "%change",
              "KS_D", "KS_p", "Welch_p"))

  for (tn in tensors) {
    v0 <- first$df$l2_norm[first$df$tensor == tn]
    v1 <- last$df$l2_norm [last$df$tensor  == tn]
    if (length(v0) < 3 || length(v1) < 3) next
    m0 <- mean(v0, na.rm = TRUE); m1 <- mean(v1, na.rm = TRUE)
    pct <- 100 * (m1 - m0) / pmax(abs(m0), 1e-12)
    ks  <- suppressWarnings(ks.test(v0, v1))
    tt  <- tryCatch(suppressWarnings(t.test(v0, v1)),
                    error = function(e) NULL)
    tp  <- if (is.null(tt)) NA_real_ else tt$p.value
    cat(sprintf("  %-10s  %12s  %12s  %+11.2f%%  %10.3f  %12.3g  %10.3g\n",
                tn, fmt(m0, 4), fmt(m1, 4), pct,
                ks$statistic, ks$p.value, tp))
  }

  cat("\n  Interpretation:\n")
  cat("  * KS_p < 0.05 → the l2 DISTRIBUTION shifted, not just the mean.\n")
  cat("  * Welch_p < 0.05 → the mean shift is statistically reliable.\n")
  cat("  * For learning, expect: a_w2 / c_w2 (output layers) to show\n")
  cat("    the largest %change. If only bias tensors (b1/b2) shift,\n")
  cat("    the network is collapsing biases toward zero rather than\n")
  cat("    learning useful weights.\n")
}

# ── 3) Bias decay ─────────────────────────────────────────────────

hdr("3) BIAS-DECAY DIAGNOSTIC")

bias_tensors <- grep("_b[12]$", tensors, value = TRUE)
cat("Biases initialise to zero; they pick up small (~1e-4) values\n")
cat("from a handful of Adam updates and decay toward zero over time.\n")
cat("If you see a clean monotonic decline, the optimizer is regularising\n")
cat("biases — fine in isolation but indicates that the only thing\n")
cat("learning is the bias channel. (Compare to the weight tensors!)\n\n")

for (tn in bias_tensors) {
  cat(sprintf("%-10s", tn))
  for (p in probes) {
    v <- p$df$l2_norm[p$df$tensor == tn]
    cat(sprintf("  %-9s=%-9s",
                p$label, fmt(mean(v, na.rm = TRUE), 6)))
  }
  cat("\n")
}

# ── 4) Output-layer magnitude bound (latest milestone) ────────────

hdr("4) OUTPUT-LAYER MAGNITUDE BOUND (latest milestone)")

latest <- probes[[length(probes)]]
ldf   <- latest$df

bound_for <- function(label, w_name, b_name, hidden_dim, n_outputs) {
  w <- ldf[ldf$tensor == w_name, ]
  b <- ldf[ldf$tensor == b_name, ]
  if (nrow(w) == 0 || nrow(b) == 0) return()
  sub(sprintf("%s — %s / %s", label, w_name, b_name))
  kv("w l2_norm mean",     fmt(mean(w$l2_norm, na.rm = TRUE), 4))
  kv("w l2_norm max",      fmt(max (w$l2_norm, na.rm = TRUE), 4))
  kv("w element_std mean", fmt(mean(w$std,    na.rm = TRUE), 4))
  kv("w |mean| mean",      fmt(mean(abs(w$mean), na.rm = TRUE), 6))
  kv("b l2_norm mean",     fmt(mean(b$l2_norm, na.rm = TRUE), 6))
  kv("b |mean| mean",      fmt(mean(abs(b$mean), na.rm = TRUE), 6))

  # Napkin estimate of pre-tanh output magnitude:
  #   pre_tanh ≈ Σ_j w_ij · h_j + b_i
  #   with h_j ReLU-like (~U(0, ~1), mean ~0.5) and w_ij ~ N(0, σ_w²)
  #   |pre_tanh| ≈ sqrt(hidden_dim) · σ_w · E[|h|]  + |b|
  sw <- mean(w$std, na.rm = TRUE)
  sb <- mean(abs(b$mean), na.rm = TRUE)
  pre_tanh_est <- sqrt(hidden_dim) * sw * 0.5 + sb
  kv("napkin |pre-tanh| estimate", fmt(pre_tanh_est, 4))
  kv("tanh of that",                fmt(tanh(pre_tanh_est), 4))
  if (tanh(pre_tanh_est) < 0.05) {
    cat("  → Policy μ is structurally pinned near zero.\n")
  } else if (tanh(pre_tanh_est) > 0.5) {
    cat("  → Policy μ has meaningful magnitude — checks out.\n")
  } else {
    cat("  → Policy μ is small but non-trivial.\n")
  }
}

# Defaults match intelligence_level_herbivore_1.rs constants:
#   HIDDEN = 64, HEAD_HIDDEN = 16, ACTOR_OUT = 4
bound_for("Actor output",  "a_w2", "a_b2", 16, 4)
bound_for("Critic output", "c_w2", "c_b2", 16, 1)
bound_for("Hidden→head",   "a_w1", "a_b1", 64, 16)
bound_for("Backbone→hidden","bk_w2","bk_b2", 64, 64)

# ── 5) Within-milestone weight ↔ predation correlation ───────────

hdr("5) ρ(per-organism weight statistic, predations) — within milestone")

cat("Per-tensor Spearman correlations between (l2_norm, std, |mean|)\n")
cat("and the agent's cumulative predation count. Positive ρ on actor\n")
cat("output weights would mean 'the more those weights have moved\n")
cat("off-zero, the more this individual ate' — i.e. learning is\n")
cat("contributing to feeding success rather than being noise.\n")

correlate_within <- function(p, c_df, tn) {
  if (is.null(c_df) || !("entity_id" %in% names(p$df))) return(NULL)
  if (!all(c("entity_id", "predations") %in% names(c_df))) return(NULL)
  c_h <- c_df[c_df$trophic_class == "Herbivore", c("entity_id", "predations")]
  c_h$predations <- as.numeric(c_h$predations)
  sub_p <- p$df[p$df$tensor == tn, c("entity_id", "l2_norm", "std", "mean")]
  j <- merge(sub_p, c_h, by = "entity_id")
  if (nrow(j) < 5) return(NULL)
  do_rho <- function(x) {
    if (var(x, na.rm = TRUE) < 1e-18 || var(j$predations, na.rm = TRUE) < 1e-18)
      return(c(NA_real_, NA_real_))
    test <- tryCatch(suppressWarnings(cor.test(x, j$predations, method = "spearman")),
                     error = function(e) NULL)
    if (is.null(test)) c(NA_real_, NA_real_)
    else c(as.numeric(test$estimate), test$p.value)
  }
  r1 <- do_rho(j$l2_norm)
  r2 <- do_rho(j$std)
  r3 <- do_rho(abs(j$mean))
  list(n = nrow(j),
       rho_l2 = r1[1], p_l2 = r1[2],
       rho_std = r2[1], p_std = r2[2],
       rho_absmean = r3[1], p_absmean = r3[2])
}

for (i in seq_along(probes)) {
  p   <- probes[[i]]
  cdf <- colony[[i]]
  if (is.null(cdf)) next
  sub(sprintf("%s — within-milestone correlations", p$label))
  cat(sprintf("  %-10s  %5s  %8s %10s  %8s %10s  %8s %10s\n",
              "tensor", "n",
              "ρ(l2)",     "p(l2)",
              "ρ(std)",    "p(std)",
              "ρ(|mean|)", "p(|mean|)"))
  found_any <- FALSE
  for (tn in tensors) {
    r <- correlate_within(p, cdf, tn)
    if (is.null(r)) next
    found_any <- TRUE
    cat(sprintf("  %-10s  %5d  %+8.3f %10.3g  %+8.3f %10.3g  %+8.3f %10.3g\n",
                tn, r$n,
                r$rho_l2, r$p_l2,
                r$rho_std, r$p_std,
                r$rho_absmean, r$p_absmean))
  }
  if (!found_any) cat("  (no matching entity_ids between probe and colony)\n")
}

# ── 6) Top vs bottom movers (actor & critic output, latest only) ─

hdr("6) TOP vs BOTTOM MOVERS — latest milestone, a_w2 & c_w2")

latest_p <- probes[[length(probes)]]
latest_c <- colony [[length(colony)]]

top_vs_bottom <- function(tensor_name) {
  if (is.null(latest_c)) { cat("  (no colony CSV available)\n"); return() }
  sub(sprintf("%s — top vs bottom by |mean|", tensor_name))
  c_h <- latest_c[latest_c$trophic_class == "Herbivore",
                  c("entity_id", "predations", "reproductions")]
  c_h$predations    <- as.numeric(c_h$predations)
  c_h$reproductions <- as.numeric(c_h$reproductions)
  sub_p <- latest_p$df[latest_p$df$tensor == tensor_name,
                       c("entity_id", "l2_norm", "std", "mean")]
  j <- merge(sub_p, c_h, by = "entity_id")
  if (nrow(j) < 10) { cat(sprintf("  (only %d matches, skip)\n", nrow(j))); return() }
  j$absmean <- abs(j$mean)
  j <- j[order(-j$absmean), ]
  k <- max(5, nrow(j) %/% 5)
  top <- j[seq_len(k), ]; bot <- j[(nrow(j) - k + 1):nrow(j), ]
  kv("k (per group)", k)
  kv("top |mean| range",
     sprintf("[%s, %s]",
             fmt(min(top$absmean), 4), fmt(max(top$absmean), 4)))
  kv("bottom |mean| range",
     sprintf("[%s, %s]",
             fmt(min(bot$absmean), 4), fmt(max(bot$absmean), 4)))
  kv("top mean predations",    fmt(mean(top$predations,    na.rm = TRUE), 2))
  kv("bottom mean predations", fmt(mean(bot$predations,    na.rm = TRUE), 2))
  kv("top mean reproductions", fmt(mean(top$reproductions, na.rm = TRUE), 3))
  kv("bottom mean reproductions",fmt(mean(bot$reproductions, na.rm = TRUE), 3))
  wt <- tryCatch(suppressWarnings(wilcox.test(top$predations, bot$predations)),
                 error = function(e) NULL)
  if (!is.null(wt)) kv("Wilcoxon U on predations", sprintf("p = %.3g", wt$p.value))
}

top_vs_bottom("a_w2")
top_vs_bottom("c_w2")

# ── 7) Per-tensor scatter summary (latest only) ───────────────────

hdr("7) SUMMARY OF LATEST-MILESTONE TENSORS")

cat(sprintf("All numerics computed on %s\n\n", latest_p$label))
cat(sprintf("  %-10s  %5s  %12s  %12s  %12s  %12s\n",
            "tensor", "n", "mean(l2)", "sd(l2)", "min(l2)", "max(l2)"))
for (tn in tensors) {
  sub_df <- latest_p$df[latest_p$df$tensor == tn, ]
  cat(sprintf("  %-10s  %5d  %12s  %12s  %12s  %12s\n",
              tn, nrow(sub_df),
              fmt(mean(sub_df$l2_norm, na.rm = TRUE), 4),
              fmt(sd  (sub_df$l2_norm, na.rm = TRUE), 4),
              fmt(min (sub_df$l2_norm, na.rm = TRUE), 4),
              fmt(max (sub_df$l2_norm, na.rm = TRUE), 4)))
}

cat("\n=== END ===\n")
sink(type = "output")
close(sink_file)
on.exit()
message(sprintf("wrote %s", output_path))
