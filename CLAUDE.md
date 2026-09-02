# FeatureGraph research

The research record: studies, papers, reproduction scripts, and the notebooks
behind them. The framework lives in a separate repository.

## The package is `featuregraph_research`

This repository ships a Python package importable as **`featuregraph_research`**,
distributed as `featuregraph-research`. The framework in
`featuregraph/featuregraph` ships `featuregraph`. They are different packages
and both can now be installed into one environment.

They were not always distinguishable. Both once declared `name = "featuregraph"`
at the same version with the same description, so pip treated them as one
distribution and installing either clobbered the other. That collision produced
a wrong docs link across the live marketing site and a false bug report before
it was found.

This package began as a fork of the framework's beta and has diverged: the same
nineteen modules, all modified, plus `utils/_arc_agi.py`, `utils/_arc_agi_pairs.py`,
`utils/_array_axes.py`, `operators/registry.py`, `operators/array_events.py`,
`behaviors/spatial_sketch.py` and `datasets/_arc_agi.py`, none of which exist
in the framework.

Two things deliberately kept the old name, and should keep it:

- **`"featuregraph"` as a representation label** in the RL and TEP experiments —
  `Literal["raw", "raw_history", "featuregraph", "augmented"]` and the dict keys
  beside it. These name a scientific result column, not a module. Renaming them
  would silently change what published tables mean.
- **`~/.cache/featuregraph/`** as the dataset cache. Renaming it would orphan
  every downloaded BIDMC and Tennessee Eastman file, and sharing the cache with
  the framework is a feature.

The prose in `README.md` and `docs/` documents the *framework's* released beta —
that is why it says `pip install "featuregraph @ git+...featuregraph.git@v0.1.0b1"`
and `import featuregraph as fg`. That is correct and was left alone. This
repository has no release tag of its own.

## What this repository is for

Studies are evidence, and evidence is only worth what its provenance is worth.

- **Contracts are frozen before the held-out run.** `artifacts/studies/` holds
  the frozen ones. Published fingerprints are claims cited elsewhere — in the
  Zenodo record, in the scientific API — so do not rewrite one to tidy it up.
  Changing a contract means a new version and a new approval, deliberately.
- **Corpora are not vendored.** Reference them by revision and take a path.
  `scripts/arc_agi_determinacy.py` records the ARC-AGI-2 commit it ran against;
  do the same for anything new.
- **Exclusions stay visible.** Records that fail a prespecified check are
  reported in the coverage table, not removed and not fixed with record-specific
  parameters.

## Findings to not re-derive

**Unmatched objects are not errors.** In the BIDMC multiscale study, respiratory
objects constructed at W=79 but absent at W=100 turned out to be phase-locked to
the ECG at cardiac frequency — a cardiogenic component, not noise. A filter that
suppressed short or "spurious-looking" occurrences would have deleted the
result. `artifacts/studies/bidmc_subject13_multiscale/report.md` has the case.

**Determinacy and correctness are different measurements.**
`artifacts/studies/arc_agi_2_determinacy/` reports what ARC-AGI-2 training pairs
pin down under a six-operator vocabulary: 29 of 1,000 training tasks fully
determined, all 29 correct on held-out test, zero determined-and-wrong, and
exactly one underdetermined key in the whole split. Coverage is a property of
the vocabulary; the precision result is what to re-measure whenever the
vocabulary grows.

**Refusal is the point.** `OperatorEvidence` distinguishes *unobserved*
(nothing was seen), *contradicted* (everything was eliminated) and
*underdetermined* (several survived). The solvers still raise; the evidence
functions report. Do not collapse those three into one error.

## Working conventions

- Topical branch per change, PR into `main`. Do not push to `main`.
- `.venv/bin/python -m pytest -q` before a PR.
- `ruff check` on files you touched.
- Reports follow `artifacts/studies/<name>/report.md` plus a CSV of per-item
  results, so a reader can check any single row rather than trusting the
  aggregate.
