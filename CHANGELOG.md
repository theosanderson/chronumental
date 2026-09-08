# Changelog

## Unreleased

Results from this version will differ from earlier ones. Every change to a
default has a flag that restores the old behaviour where one is listed.

### Changed defaults

- The clock rate is held at its estimate rather than fitted jointly with the
  branch durations, which biased it low. `--floating_clock_rate` restores the
  free fit; `--enforce_exact_clock` is removed because it is now the default.
- The starting clock rate comes from a Theil-Sen root-to-tip regression
  rather than least squares, so a few tips with extreme dates or divergences
  no longer drag the slope.
- The fit starts from the tip dates: each tip on its own date and each
  internal node placed from its children. `--initialise clock` restores the
  old start from the clock rate alone.
- `--variance_dates` defaults to 3 days rather than 0.3, and is combined in
  quadrature with the precision window of month- and year-only dates rather
  than multiplied by it. `--multiply_date_precision` restores the product.
- The prior on the root date is a Cauchy with a 100-year scale
  (`--root_date_prior_scale_days`) rather than a Normal with a 1000-day one,
  which had pulled deep roots forward.
- A branch length on the root is ignored: a root has no parent for time to
  elapse from. The output tree writes it as zero.
- Conversions between days and years use 365.25, and a date given as a
  month or a year is centred in that period, with the period's true length
  as its window.
- Fitting stops once the predicted node dates and the clock rate have
  settled (`--convergence_tol_days`, `--convergence_rate_tol`,
  `--convergence_patience`, `--convergence_check_every`), and `--steps` is a
  ceiling. `--disable_early_stopping` runs the full count.

### Added

- `--clock_filter_iqd` discards the dates of tips whose root-to-tip
  divergence sits too far from the clock line. Off by default.

### Removed

- The horseshoe model, and with it `--model`, `--initial_tau` and
  `--hs_scale`.
- `--enforce_exact_clock`, see above.

### Performance

- Root-to-tip sums are computed by pointer jumping over a parent-index array
  rather than a sparse matrix, cutting the largest allocation from 2.6 GB to
  80 MB on a 300k-tip tree.
- SVI steps run in chunks of ten inside a jitted `lax.scan`, with the
  best-loss parameters tracked on device. Roughly ten times faster per step.
- The input tree is parsed once rather than twice, and the traversal is
  iterative so deep trees no longer overflow the recursion limit.

### Fixed

- Import on current jax (`jax.lib.xla_bridge` was removed) and the
  `TruncatedNormal` argument order under numpyro 0.9 and later, which had
  broken `--variance_on_clock_rate`.
- `--always_use_final_params` raised `UnboundLocalError`.
- The best-loss parameters were paired with the loss of the previous step,
  so a diverging fit kept its exploded parameters instead of falling back.
- A zero or NaN clock estimate now fails with a message instead of producing
  an infinite tree.
- Duplicate node labels, including a genuine `NODE_0000001` alongside an
  unlabelled node, are rejected instead of silently sharing parameters.
- Interrupting a run without a terminal now saves the results so far.

### Packaging

- Python 3.11 or later is required. Dependencies are declared at the oldest
  versions that install and run together.
- The version is derived from git tags by `setuptools_scm` and written to
  `_version.py`; the license is declared as an SPDX expression.
