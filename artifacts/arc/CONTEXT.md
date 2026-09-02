# ARC-AGI-2 research context

State of play and decision log for the ARC-AGI-2 line. Written so that a
session starting cold can pick this up without the conversation that produced
it. The two study records (`README.md`, `region_objects_scope.md`) carry the
measurements; this file carries the reasoning, the corrections, and the open
questions.

Branch: `claude/arc-agi-2-discussion-nxn1nx`.
Dataset: ARC-AGI-2 revision `f3283f727488ad98fe575ea6a5ac981e4a188e49`.

## What this line is

**A representation study, not a solver.** The question is not "was the task
solved" but "where does a declared vocabulary apply, and where does it run
out". A solver returns one bit per task; the object tables return a status per
block and per region, so a task that is two-thirds describable is recorded as
two-thirds describable.

This was a deliberate fork. The work began as a solver, and the solver framing
was abandoned after the evaluation split turned out not to contain the task
family the training split teaches.

## How it got here

**The original approach** (`featuregraph/utils/_arc_agi.py`, still present)
inferred a block-operator layout and predicted test outputs. Its evaluation
notebook built a failure taxonomy by substring-matching `ValueError` messages,
which meant it stopped at the first bad block and could only ever report
supported/unsupported.

**The fork.** Looking at the evaluation set properly showed it was not harder
versions of the training tasks — it required composing the tools the training
set supplies, not applying them in learned form. Measurement confirmed this
more sharply than expected: the non-trivial tiling family is **entirely absent**
from the evaluation split.

**Block composition** was then rebuilt as `BlockComposition`, a `Behavior`
producing one object per output block, retaining full candidate sets.

**Regions** followed, because block composition reaches 5.1% of training and
0% of evaluation, while same-shape tasks are two thirds of both splits.

## Findings

### Block composition (`README.md`, `describability.csv`)

| | training (1000) | evaluation (120) |
| --- | ---: | ---: |
| output tiles the input | 731 (73.1%) | 81 (67.5%) |
| …of which 1×1, same shape | 680 | 81 |
| **non-trivial tiling tasks** | **51 (5.1%)** | **0 (0.0%)** |
| demonstration blocks | 1075 | – |
| blocks described (≥1 operator) | 754 (70.1%) | – |
| blocks ambiguous (>1 operator) | 97 | – |
| layout determined by block position | 21 tasks | – |
| layout determined by input state | 2 tasks | – |

State-layout-only tasks: `007bbfb7`, `5b6cbef5`. Out-of-frame reasons:
`output_not_block_multiple` (256 / 39), `inconsistent_block_layout` (13 / 0).

### Regions (`region_objects_scope.md`, `region_probe.csv`)

Over the 761 same-shape tasks:

| | training (680) | evaluation (81) |
| --- | ---: | ---: |
| background mask preserved | 211 (31.0%) | 38 (46.9%) |
| output colour determined by simple properties | 10 (1.5%) | 0 (0.0%) |
| **edit components inside one input part (4-conn)** | **94.2%** | **94.0%** |
| same, 8-connectivity | 91.7% | 90.5% |
| tasks with every edit aligned | 545 (80.1%) | 62 (76.5%) |

Median regions per input grid: 2 / 1. Maximum: 86 / 360.

### The comparison that matters

| | training | evaluation |
| --- | ---: | ---: |
| block composition, non-trivial tiling | 5.1% | 0.0% |
| region construction, edits aligned | 94.2% | 94.0% |

Block composition collapses between splits. The region construction does not.
That cross-split stability is the property a representational claim needs.

## Design decisions worth not relitigating

**The two solver families are two groupings.** `fixed_layout_training_cycle`
and `state_training_cycle` in the old module are not two algorithms. They are
`resolve_layout(by=("block_row", "block_column"))` and
`resolve_layout(by=("block_state",))` over one object table. A third hypothesis
is a third grouping, not a third solver.

**Candidate sets are retained, never collapsed.** A block matched by no
operator is an object with an empty candidate set, not an exception. This is
what makes partial describability reportable.

**Constructions declare their parameters.** Connectivity, background colour,
region definition, background inclusion, operator vocabulary — all recorded in
`objects.construction`, in the same sense `Oscillation` records `smooth_window`
and `diff_lag`. Different declarations produce different objects from identical
observations.

**Object boundary and predictability are separate questions.** Regions are the
right boundary (94% of edits respect them) *and* region edits are not
predictable from simple properties (1.5% of tasks). Both true at once. This
distinction carries the whole representation-not-solver argument.

## Corrections made along the way

Recorded because each changed a decision:

1. **`007bbfb7` intersection.** Claimed that intersecting across pairs collapses
   `{copy, flip_horizontal}` to `copy`. It does not — it empties out entirely,
   because that task's layout depends on input contents. This produced the
   groupings insight.
2. **Shape-invalid quarter turns.** Dropping them rather than masking them
   misclassified **149 tasks** as out of frame, because some inputs were square
   and others were not, so vocabularies were not comparable across pairs.
3. **"Regions might describe most of the benchmark."** Right about the object
   boundary, wrong about describability. The probe separated the two claims.
4. **`touches_border`** measured grid extent from surviving cells, so interior
   regions read as touching the edge when background objects were excluded.
5. **Test outputs.** Public ARC-AGI-2 files include test outputs, so
   `no_output_grid` never fires on this dataset. Layout inference therefore
   filters `pair_type == "train"` explicitly rather than relying on absence.

## Paper positioning (discussed, concluded)

**Not an ARC paper.** The decisive argument: a scratch NumPy prototype produced
the identical describability numbers before `composition.py` existed. If the
framework is not necessary to produce the finding, the finding cannot be
evidence for the framework. Supporting reasons: nobody claimed block
composition solves ARC-AGI-2, the vocabulary is arbitrary, and ARC Prize
designed v2 to defeat exactly this and said so.

**Retaining candidate sets and intersecting them is version-space learning** —
Mitchell's candidate elimination, in the degenerate flat-hypothesis-space case
where it reduces to set intersection. Presenting it as a table operation is an
exposition and engineering contribution, not a learning-theory one. State it
that way before a reviewer does.

**Where it could go.** The framework paper's cross-domain section rests on
BIDMC and TEP, which differ physically but are *formally identical*: both 1-D
ordered signals through `Oscillation`, same schema. That is one demonstration
performed twice, and it is the softest part of the paper. ARC is the first
formally different setting — 2-D, discrete, no time axis, different behavior
type, different identity rule — so it tests the four claimed representational
invariants rather than re-instancing them.

**But not in the current paper.** Its abstract opens on time-series and its
framing is temporal throughout; a 2-D discrete demonstration would make it
incoherent. Its own conclusion names "broader behavioral types" as future work.
So: successor paper, or a reframed v2.

**Connective thread if it is written:** coverage reporting. The abstract
already highlights "explicit coverage failure rather than superior fault
prediction" (TEP Fault 6 producing no complete post-response objects). ARC
evaluation at exactly zero coverage is the extreme instance of the same idea.
ARC is a supporting case, never the headline.

## Open questions

- **Stage 2 change taxonomy.** `unchanged` / `recoloured` / `partially_changed`
  / `erased` is a guess made before the region object table existed. It can now
  be designed against real regions. Not built.
- **Stage 3**, per-region describability measurement. Not built.
- **`inconsistent_block_layout`** (13 training tasks) conflates genuinely
  out-of-frame tasks with tasks whose layout may scale with the input. Under
  the current design that would be another grouping, not another solver.
- **Repo scope.** The README says the beta line extends the empirical reach of
  oscillation and accumulation, and that successor object models belong on
  `featuregraph/featuregraph` `main`. `BlockComposition` and `RegionObjects`
  are new object models. Putting them on this research branch sidesteps that
  rule rather than settling it.
- **`notebooks/arc-agi_evaluation.ipynb`** is superseded — it is the
  exception-substring-matching version. Left in place as a research record.

## What is deliberately not being built

An operator vocabulary for region edits, chasing coverage upward. That is the
solver trap this line already walked into once. **Coverage is the measurement,
not the target.** If the honest number is low, the honest number is reported.

## Files

| path | what |
| --- | --- |
| `src/featuregraph/behaviors/composition.py` | `BlockComposition`, `resolve_layout` |
| `src/featuregraph/behaviors/regions.py` | `RegionObjects`, `edit_alignment` |
| `src/featuregraph/operators/spatial.py` | block operator vocabulary as `OperatorRecord`s |
| `src/featuregraph/operators/grids.py` | grid reconstruction, region labelling, hole counting |
| `src/featuregraph/utils/_arc_agi.py` | original solver, still exported |
| `src/featuregraph/datasets/_arc_agi.py` | cell-level observation loader, caches downloads |
| `experiments/arc/describability.py` | block describability scan |
| `experiments/arc/region_probe.py` | region structure and edit alignment scan |
| `notebooks/arc-agi_block_objects.ipynb` | block construction demonstration |
| `notebooks/arc-agi_region_objects.ipynb` | region construction demonstration |
| `notebooks/arc-agi_exercise.ipynb` | five predict-then-check exercises |
| `tests/test_block_composition.py`, `test_region_objects.py`, `test_grid_operators.py`, `test_spatial_operators.py` | |

## Reproducing

```bash
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 /tmp/ARC-AGI-2
python -m pip install -e ".[dev]"

python -m experiments.arc.describability --data-dir /tmp/ARC-AGI-2/data
python -m experiments.arc.region_probe --data-dir /tmp/ARC-AGI-2/data
python -m pytest
```

Both experiments run through the library, so the recorded artifacts and the
constructions cannot drift apart. The region probe originally carried its own
region labelling; when it was moved onto the library the refactor reproduced
the earlier table row for row across all 1,522 rows and 15 columns, so both
implementations agreed before one was removed.
