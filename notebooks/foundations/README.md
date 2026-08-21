# FeatureGraph Foundations

This directory separates maintained foundations from historical research
records.

FeatureGraph's maintained foundation is not a general mathematics or machine
learning curriculum. It is the smaller sequence of representations needed to
construct explicit, deterministic behavioral objects:

```text
observations
    → coordinates and states
    → boundaries and transitions
    → bounded objects
    → named operators and provenance
    → measurements and composed outputs
```

## Maintained tracks

| Track | Central contract | Guide |
|---|---|---|
| Array and spatial representation | Make every cell's position, grouping, and local coordinates explicit. | [`curated/01_array_and_spatial_representation.md`](curated/01_array_and_spatial_representation.md) |
| Temporal and behavioral representation | Turn primitive states and transitions into bounded objects. | [`curated/02_temporal_and_behavioral_representation.md`](curated/02_temporal_and_behavioral_representation.md) |
| Operators, provenance, and instructions | Name transformations, validate contracts, and preserve how each result was derived. | [`curated/03_operators_provenance_and_instructions.md`](curated/03_operators_provenance_and_instructions.md) |

These guides point to tested library operations. They are the maintained entry
point for new work.

## Historical source records

The 12 migrated collections and precursor projects are preserved under
[`archive/source_records/`](archive/source_records/). They are provenance, not
supported tutorials or importable packages.

Every source artifact has a disposition in
[`archive/INVENTORY.csv`](archive/INVENTORY.csv):

- `promote`: its idea has been incorporated into a maintained guide or tested
  operation.
- `reference`: it remains useful supporting material.
- `archive`: it records superseded, duplicate, incomplete, or dependency-heavy
  exploration.
- `remove`: it adds no substantive content to a future derivative, but remains
  here so the migration record is complete.

Regenerate the inventory from the repository root with:

```bash
python scripts/audit_foundations.py
```

## Promotion rule

Exploration remains historical until it has:

1. a stated input representation;
2. a named operator;
3. a stated output representation;
4. explicit invariants;
5. deterministic validation;
6. evidence that the operation repeats across problems.

Only then should the operation move into `src/featuregraph/` and receive a
focused test.
