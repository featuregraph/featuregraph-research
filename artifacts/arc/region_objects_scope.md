# Region objects: scope

Scoping document for the construction that follows block composition.

## Status

Stage 1 and the alignment measurement are built:
`featuregraph.behaviors.regions.RegionObjects` and `edit_alignment`, with
`experiments/arc/region_probe.py` running both. The probe originally carried its
own region labelling; it now runs on the library, and the refactor reproduced
the earlier table row for row across all 1,522 rows and 15 columns, so the two
implementations agreed before one was removed.

Stage 2's change taxonomy and Stage 3 are not built.

## Why this construction

Block composition described 5.1% of the ARC-AGI-2 training split and 0.0% of
evaluation. Same-shape tasks — output dimensions equal to input dimensions —
are 68.0% of training and 67.5% of evaluation, and block decomposition says
nothing about them: their block layout is 1×1, which asserts nothing.

If the framework's representational claim is to be tested on a population that
matters, it has to be tested there.

## What the probe established

`experiments/arc/region_probe.py`, run over all 761 same-shape tasks
(ARC-AGI-2 revision `f3283f727488ad98fe575ea6a5ac981e4a188e49`):

| | training (680) | evaluation (81) |
| --- | ---: | ---: |
| median regions per input grid | 2 | 1 |
| maximum regions in one grid | 86 | 360 |
| background mask preserved across demos | 211 (31.0%) | 38 (46.9%) |
| output colour a function of simple region properties | 10 (1.5%) | 0 (0.0%) |
| **edit components lying inside one input part** | **94.2%** | **94.0%** |
| tasks where every edit is region-aligned | 545 (80.1%) | 62 (76.5%) |

Under 8-connectivity instead of 4: alignment 91.7% / 90.5%, fully-aligned tasks
534 / 60. The finding is not an artefact of the connectivity choice.

### Two claims that must stay separate

**Regions are the right object boundary.** 94% of all edit components sit
entirely within a single input part — a foreground component or a background
component. Changes respect region structure.

**Region edits are not predictable from simple region properties.** Only 10
training tasks and 0 evaluation tasks have an output colour determined by input
colour, size, size rank, or bounding box. The obvious recolour vocabulary is
dead on arrival.

These are different claims, and only the first is needed for the
representational result. An earlier statement in this research line — that
regions "might describe most of the benchmark" — conflated them and was wrong
in the second sense.

### The number that makes this worth building

Alignment is 94.2% in training and 94.0% in evaluation. Block composition was
5.1% and 0.0%. **The region construction transfers across splits where block
composition did not.** That is the property a cross-domain representational
claim actually needs, and it is the reason to spend effort here rather than on
extending the block operator vocabulary.

## Proposed construction

### Stage 1 — `RegionObjects` (the deliverable that carries the claim)

A `Behavior` producing one object per connected region of a single grid.
Depends on nothing else and works on every grid in both splits, whether or not
its task is describable.

```text
grid cell observations
    → background/foreground states
    → connected-component identities
    → one row per region object
    → computational queries
```

- `add_primitives` — `is_background` from the declared background colour.
- `add_ids` — `region_id` from connected-component labelling.
- `add_features` — per-region transforms: size, bounding box, fill ratio,
  colour, hole count, border contact, centroid.
- `summarize` — one row per region.

Declared construction parameters, in the same sense as `Oscillation`'s
`smooth_window` and `diff_lag`: connectivity (4 or 8), background colour, and
whether a region is a uniform-colour component or a non-background component.
The describability figures are conditional on these, so sensitivity to each
gets measured rather than asserted.

### Stage 2 — `RegionCorrespondence` (child construction)

A second behavior constructed inside the intervals its parent supplies, the
same relationship wave-derived accumulation has to oscillation. Given input
regions and an output grid, each correspondence object records the region's
fate: `unchanged`, `recoloured`, `partially_changed`, `erased`, plus edits
attributable to no input region.

The measurable is edit alignment and per-region change classification coverage —
not an operator that predicts the edit.

### Stage 3 — describability measurement

Reuses the `describability.py` harness, reporting per region rather than per
task, split by connectivity and background colour.

## Explicitly out of scope

**Building an operator vocabulary for region edits and chasing coverage
upward.** That is the solver trap that this research line already walked into
once. Coverage is the measurement, not the target. If the honest number is low,
the honest number gets reported.

## Risks

- **Region-count tail.** One evaluation grid yields 360 regions under
  4-connectivity, one training grid 86. A per-region object table handles this,
  but per-task cost grows and the describability scan will be slower than the
  block scan.
- **Conditional numbers.** Every figure depends on connectivity, background
  colour, and region definition. Sensitivity to all three must be reported, and
  the 8-connectivity check above is only the first of them.
- **Background colour is assumed to be 0.** Some tasks use a different dominant
  colour. This needs to be a declared parameter with measured sensitivity, as
  it now is for block composition.
- **Stage 2 has genuine design uncertainty.** The change taxonomy above is a
  guess and will likely need revision once the object table exists. Stage 1
  does not depend on getting it right.

## Recommendation

Build Stage 1 and the alignment measurement from Stage 2. Skip the edit
vocabulary entirely.

That yields a region object table over two thirds of both ARC-AGI-2 splits,
with retained provenance and coverage reported as data — which is what a
cross-domain representational claim requires — without re-entering solver
territory. Stage 3 follows only if Stage 1 holds up.

## Reproduce

```bash
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 /tmp/ARC-AGI-2
python -m experiments.arc.region_probe \
    --data-dir /tmp/ARC-AGI-2/data \
    --describability artifacts/arc/describability.csv \
    --output artifacts/arc/region_probe.csv
```
