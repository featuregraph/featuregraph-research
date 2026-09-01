# ARC-AGI-2 block-composition describability

A representation study, not a solver benchmark.

FeatureGraph's claim is that declaring what data contains, and constructing
explicit objects from that declaration, produces a traceable account of a
domain. This study asks whether that construction carries from one-dimensional
observation sequences to two-dimensional discrete grids, and what it reports
when it reaches the edge of what it can describe.

## Question

Given a small, explicitly declared operator vocabulary, **where does it
apply across ARC-AGI-2, and where does it run out?**

That is deliberately narrower than "was the task solved". A solver returns one
bit per task. The block-object table returns a status per block, so a task that
is two-thirds describable is recorded as two-thirds describable.

## Construction

`featuregraph.behaviors.composition.BlockComposition` builds one object per
output block of a grid pair:

```text
grid cell observations
    → per-cell operator match states
    → block boundaries and identities
    → one row per block object
    → computational queries
```

The declared vocabulary is seven operators — `copy`, `flip_horizontal`,
`flip_vertical`, `rotate_90`, `rotate_180`, `rotate_270`, `background` — held
as `OperatorRecord` values so the vocabulary carries its own provenance. A
quarter turn is only shape valid on a square grid, and is masked rather than
dropped so that pairs stay comparable.

Every block keeps its **full candidate set**. A block matched by no operator is
recorded, not raised. Resolving an instruction layout is then an intersection
over a chosen grouping of that table, and the grouping is the hypothesis class:

| grouping | hypothesis |
| --- | --- |
| `by=("block_row", "block_column")` | layout fixed to block position |
| `by=("block_state",)` | layout derived from the corresponding input cell |

These are not two solvers. They are two `GROUP BY` clauses over one object
table.

Layouts are resolved from demonstration pairs only; test pairs are described
but never intersected.

## Result

Full public ARC-AGI-2, revision `f3283f727488ad98fe575ea6a5ac981e4a188e49`:

| | training (1000) | evaluation (120) |
| --- | ---: | ---: |
| output tiles the input | 731 (73.1%) | 81 (67.5%) |
| …of which 1×1, same shape | 680 | 81 |
| **non-trivial tiling tasks** | **51 (5.1%)** | **0 (0.0%)** |
| demonstration blocks | 1075 | – |
| blocks described (≥1 operator) | 754 (70.1%) | – |
| blocks unmatched | 321 (29.9%) | – |
| blocks ambiguous (>1 operator) | 97 | – |
| layout determined by block position | 21 tasks | – |
| layout determined by input state | 2 tasks | – |
| determined by either | 23 tasks | – |

Out-of-frame reasons: `output_not_block_multiple` (256 training, 39
evaluation) and `inconsistent_block_layout` (13 training).

## Reading

**The block-composition family is absent from the evaluation split.** Not
rarer, not harder — absent. Every evaluation task that nominally sits in frame
is same-shape, where a 1×1 block layout asserts nothing about the task. The
tiling structure that 51 training tasks exhibit appears in none of the 120
evaluation tasks.

This is ARC-AGI-2 behaving as designed. The benchmark was built so that a
vocabulary induced from the training split does not transfer to evaluation by
being applied in the form it was learned. A study that reported only training
accuracy would have missed this entirely; the split-wise describability table
makes it the headline.

Two consequences for this research line:

1. **Extending the operator vocabulary is not worth doing.** More operators
   would move the 754/1075 block figure and nothing else. There is no
   evaluation-set task for them to reach.
2. **Same-shape tasks are the real population** — 68.0% of training and 67.5%
   of evaluation. Block decomposition is the wrong object boundary for them;
   their objects are connected regions, not blocks. That is the next
   construction to build, and it is the same four-stage pipeline with a
   different identity rule.

## Honest limits

- The 1075-block denominator counts blocks per demonstration pair, so a task
  with five demonstrations contributes five times its layout size. Per-layout
  counts are recoverable from `block_rows` × `block_columns` in the CSV.
- "Describable" means at least one declared operator reproduces the block
  exactly. It does not mean the resulting layout generalizes to the test pair.
- `inconsistent_block_layout` marks tasks whose demonstrations tile to
  different layout shapes. Some are legitimately out of frame; others might be
  described by a layout that scales with the input. That distinction is not
  drawn here.
- Only background colour 0 was scanned.

## Reproduce

```bash
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 /tmp/ARC-AGI-2
python -m experiments.arc.describability \
    --data-dir /tmp/ARC-AGI-2/data \
    --output artifacts/arc/describability.csv
```

Outputs `describability.csv` (one row per task) and
`describability_summary.json` (split rollup, dataset revision, and the declared
vocabulary).
