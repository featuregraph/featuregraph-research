# ARC-AGI-2 determinacy under a declared operator vocabulary

What the training pairs determine, measured across the corpus. This is not a
solver score. A task counted as *determined* is one the training pairs fully
constrain under this operator vocabulary; whether the resulting rules are
correct is reported separately, from the held-out test pair.

- Corpus: `arcprize/ARC-AGI-2` at `f3283f727488ad98fe575ea6a5ac981e4a188e49`
- Vocabulary: `copy`, `flip_horizontal`, `flip_vertical`, `rotate_90`,
  `rotate_180`, `rotate_270`
- Reproduce: `python scripts/arc_agi_determinacy.py --corpus <arc-agi-2>/data`
- Per-task results: `task_determinacy.csv`

## Outcomes

Four conditions, and they are not the same failure. *Inapplicable* means the
task is not shaped for this construction at all — the output is not a whole
number of input-shaped blocks, or the block layout does not match the grid.
*Contradicted* means every candidate was eliminated: the pairs are shaped for
it and no operator in the vocabulary reproduces the output. *Underdetermined*
means several candidates survived every pair. *Determined* means exactly one
survived for every key.

### Training split, 1000 tasks

| | fixed-layout | state-layout |
| --- | ---: | ---: |
| determined | 26 (2.6%) | 3 (0.3%) |
| underdetermined | 0 | 0 |
| contradicted | 705 (70.5%) | 14 (1.4%) |
| inapplicable | 269 (26.9%) | 983 (98.3%) |
| **held-out correct** | **26 / 26** | **3 / 3** |

### Evaluation split, 120 tasks

| | fixed-layout | state-layout |
| --- | ---: | ---: |
| determined | 0 | 0 |
| contradicted | 81 (67.5%) | 0 |
| inapplicable | 39 (32.5%) | 120 (100%) |

## What the numbers say

**When it commits, it is right.** Twenty-nine tasks across the corpus were
fully determined, and all twenty-nine reproduce their held-out test output
exactly. No task was determined and wrong. The refusal discipline bought
complete precision at 2.6% coverage of the training split, which is the trade
it is supposed to make.

**Ambiguity is not the failure mode; contradiction is.** Exactly one
underdetermined key appeared in 1,000 training tasks. Intersecting six rigid
operators across several pairs almost always collapses to one candidate or to
none. The guard against ambiguity is real but rarely fires — this vocabulary
is too rigid to be ambiguous and too small to be sufficient.

**The two families are disjoint.** No task was determined by both. Their
applicability conditions do not overlap: the state family additionally
requires the block layout to match the grid shape, which excludes every task
the fixed family resolves.

**The evaluation split contains nothing this vocabulary reaches.** Zero
determined tasks out of 120, against 29 in the training split. The vocabulary
applies structurally to 81 of the 120 — the outputs do tile — and is
contradicted on every one of them.

## What this does not establish

Coverage here is a property of a six-operator vocabulary, not of the approach.
A larger vocabulary would move tasks from contradicted toward determined, and
would also create the opportunity for genuine underdetermination that this
vocabulary is too rigid to produce. Nothing here shows the trade holds as the
vocabulary grows; the precision result is worth re-measuring at every
extension, and is the thing to watch.

The state family uses two states, background and foreground, split on colour
zero. That is one declaration among many possible ones, and its 98.3%
inapplicability is mostly the block-layout constraint rather than a statement
about states.
