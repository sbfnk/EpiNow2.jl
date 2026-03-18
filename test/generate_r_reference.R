# Generate reference results from R's EpiNow2 for numerical validation
# Run with: Rscript test/generate_r_reference.R

library(EpiNow2)
library(data.table)

# Use first 30 days of example data for speed
data <- example_confirmed[1:30]

outdir <- file.path("test", "reference")
dir.create(outdir, showWarnings = FALSE)

# Save the input data
write.csv(data, file.path(outdir, "input_data.csv"), row.names = FALSE)

save_results <- function(res, prefix) {
  s <- summary(res, type = "parameters")
  write.csv(
    s[variable == "R"],
    file.path(outdir, paste0(prefix, "_rt.csv")), row.names = FALSE
  )
  write.csv(
    s[variable == "infections"],
    file.path(outdir, paste0(prefix, "_infections.csv")), row.names = FALSE
  )
  write.csv(
    s[variable == "reported_cases"],
    file.path(outdir, paste0(prefix, "_reports.csv")), row.names = FALSE
  )
}

# ── Test 1: Simplest model (no GT, no delay, no week effect) ─────────
cat("Running test 1: simplest model...\n")
res1 <- estimate_infections(
  data,
  generation_time = generation_time_opts(Fixed(1)),
  delays = delay_opts(Fixed(0)),
  obs = obs_opts(week_effect = FALSE),
  rt = rt_opts(),
  gp = gp_opts(basis_prop = 0.2, boundary_scale = 1.5),
  stan = stan_opts(samples = 1000, warmup = 500, chains = 2, seed = 42),
  verbose = FALSE
)
save_results(res1, "test1")

# ── Test 2: With generation time ─────────────────────────────────────
cat("Running test 2: with generation time...\n")
res2 <- estimate_infections(
  data,
  generation_time = generation_time_opts(
    LogNormal(meanlog = 1.6, sdlog = 0.5, max = 14)
  ),
  delays = delay_opts(Fixed(0)),
  obs = obs_opts(week_effect = FALSE),
  rt = rt_opts(),
  gp = gp_opts(basis_prop = 0.2, boundary_scale = 1.5),
  stan = stan_opts(samples = 1000, warmup = 500, chains = 2, seed = 42),
  verbose = FALSE
)
save_results(res2, "test2")

# ── Test 3: With generation time + reporting delay ───────────────────
cat("Running test 3: with GT + reporting delay...\n")
res3 <- estimate_infections(
  data,
  generation_time = generation_time_opts(
    LogNormal(meanlog = 1.6, sdlog = 0.5, max = 14)
  ),
  delays = delay_opts(LogNormal(meanlog = 0.5, sdlog = 0.5, max = 10)),
  obs = obs_opts(week_effect = FALSE),
  rt = rt_opts(),
  gp = gp_opts(basis_prop = 0.2, boundary_scale = 1.5),
  stan = stan_opts(samples = 1000, warmup = 500, chains = 2, seed = 42),
  verbose = FALSE
)
save_results(res3, "test3")

# ── Test 4: Full model with week effect ──────────────────────────────
cat("Running test 4: full model with week effect...\n")
res4 <- estimate_infections(
  data,
  generation_time = generation_time_opts(
    LogNormal(meanlog = 1.6, sdlog = 0.5, max = 14)
  ),
  delays = delay_opts(LogNormal(meanlog = 0.5, sdlog = 0.5, max = 10)),
  obs = obs_opts(week_effect = TRUE),
  rt = rt_opts(),
  gp = gp_opts(basis_prop = 0.2, boundary_scale = 1.5),
  stan = stan_opts(samples = 1000, warmup = 500, chains = 2, seed = 42),
  verbose = FALSE
)
save_results(res4, "test4")

cat("Done! Reference results saved to", outdir, "\n")
