# Changelog

All notable changes to FeatureGraph are documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Python package versions follow [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

### Added

- `RegionObjects` behavior constructing one object per connected region of a
  grid, with size, bounding box, fill ratio, hole count, centroid, and border
  contact. Connectivity, background colour, region definition, and background
  inclusion are declared construction parameters recorded on the result.
- `edit_alignment`, measuring whether the differences between paired grids fall
  inside single regions. This asks whether regions are the right object
  boundary, which is separate from whether an edit can be predicted.
- Shared grid operators for rebuilding a grid from cell observations, labelling
  connected regions, and counting enclosed holes, so that grid behaviors cannot
  drift apart in what they call a region.
- Region study over the 761 same-shape ARC-AGI-2 tasks: 94.2% of training and
  94.0% of evaluation edit components lie inside a single region, against 5.1%
  and 0.0% of tasks describable by block composition.

- `BlockComposition` behavior constructing one object per output block of a grid
  pair, carrying each block's full operator candidate set instead of collapsing
  it to a single answer or raising on the first block that fails.
- `resolve_layout`, which infers an instruction layout by intersecting block
  candidate sets over a chosen grouping. Grouping by block position and grouping
  by input-cell state are the two hypothesis classes previously implemented as
  separate ARC solver families.
- Declared block-operator vocabulary as `OperatorRecord` values, with the
  constant-fill operator parameterized by background colour and quarter turns
  masked rather than dropped on non-square grids.
- ARC-AGI-2 describability study measuring where the block-composition
  vocabulary applies across all 1,120 public tasks, with a per-task table, a
  split-level summary recording the dataset revision, and a study record.
- Causal FeatureGraph observation encoders and a paired-seed DQN experiment for
  raw, raw-history, behavioral-only, and augmented CartPole/MountainCar states.
- Reproducible, cached MountainCar trajectories generated from the canonical
  dynamics with a seeded exploratory momentum policy and RL transition schema.
- Deterministic, cached CartPole trajectories for oscillation and accumulation
  research, including provenance metadata and an end-to-end behavioral-object
  integration test.
- Canonical paper index and revision guidance.
- CI execution of the public alpha demonstration notebook.
- Tests for manifest-driven reproduction.

### Changed

- Replaced the orphaned `spatial_sketch` prototype with the `composition`
  behavior module, which builds the same representation through the shared
  `Behavior` construction pipeline.
- Threaded `background_color` through the ARC block-composition utilities so a
  task whose background is not zero is no longer filled with zeros.
- Defined `alpha/v0.1.x` as a living oscillation-and-accumulation research line while keeping successor architecture work on `main`.
- Pinned the Tennessee Eastman source revision and made its cache revision-specific.
- Made the reproduction script read and validate the versioned manifest.
- Focused the README on alpha installation, documentation, and research resources.
- Completed DOI metadata and converted the prerelease checklist into an alpha release record.
- Consolidated repository ignore rules.

## [0.1.0a1] - 2026-07-24

### Added

- Alpha implementation of explicit oscillation objects.
- Wave-derived accumulation objects with parent completeness propagation.
- Inspectable construction features and object tables.
- Deterministic query interface.
- BIDMC and Tennessee Eastman dataset loaders.
- Reproducibility script and manifest for paper tables and figures.
- Environment and hardware capture for benchmark runs.
- Data-download and archival-release documentation.
- Citation and Zenodo metadata.
- CI tests across Python 3.10 through 3.13.
- Package-build, clean-wheel-install, and reproduction smoke checks.

### Fixed

- Assigned peaks and troughs to the preceding sample at directional reversals.
- Derived complete oscillation boundaries from explicit trough–peak–trough extrema.
- Preserved flat regions inside extrema-defined object boundaries.
- Propagated parent oscillation completeness to accumulation summaries.
- Plotted the smoothed reactor-temperature signal used to construct Eastman boundaries.
- Corrected corrupted arrow and em-dash characters in the README.

[Unreleased]: https://github.com/featuregraph/featuregraph/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/featuregraph/featuregraph/releases/tag/v0.1.0a1
