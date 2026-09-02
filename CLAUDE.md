# FeatureGraph research

The research record: studies, papers, reproduction scripts, and the notebooks
behind them. The framework lives in a separate repository.

## Name collision — read this before debugging anything

**This repository and `featuregraph/featuregraph` both ship a Python package
called `featuregraph`.** Same import path, different modules. `utils/_arc_agi.py`
is here and not there; `contracts/` and `study_builder/` are there and not here.

- This repo publishes docs at **`featuregraph.readthedocs.io`**.
- The framework publishes at `featuregraph-framework.readthedocs.io`.

That collision has already produced a wrong docs link on the live marketing site
and a false bug report. If an import surprises you, check which package you are
in before concluding anything is broken.

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
