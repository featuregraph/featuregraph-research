# Mathematical Foundations Notebooks

This directory preserves exploratory notebooks migrated from
[`habibdraft/numpy`](https://github.com/habibdraft/numpy). The original
repository remains unchanged. The notebooks are organized here by the kind of
representation or operator they investigate.

These are research records, not polished tutorials. Existing cells and outputs
were preserved during migration.

## Research map

| Area | Central question | Directory |
|---|---|---|
| Linear algebra | How do matrices represent and transform structure? | [`01_linear_algebra/`](01_linear_algebra/) |
| Numerical methods | How can a function be represented by an approximation? | [`02_numerical_methods/`](02_numerical_methods/) |
| Probability | How do distributions and transition matrices represent uncertainty and state? | [`03_probability/`](03_probability/) |
| Learning | How do observations accumulate into estimates and decisions? | [`04_learning/`](04_learning/) |
| Neural computing | How are functions and positions represented and learned computationally? | [`05_neural_computing/`](05_neural_computing/) |
| Spatial reasoning | How can local neighborhood relations represent a grid? | [`06_spatial_reasoning/`](06_spatial_reasoning/) |

## ARC spatial-reasoning path

The notebooks form a useful progression for the ARC-AGI work:

1. [Linear algebra plotting](01_linear_algebra/linear_algebra_plotting.ipynb)
   — make matrix structure visible.
2. [Neighbors matrix](06_spatial_reasoning/neighbors_matrix.ipynb)
   — represent local spatial relations.
3. [Reflection matrices](01_linear_algebra/reflection_matrices.ipynb)
   — encode a spatial symmetry.
4. [Rotation matrices](01_linear_algebra/rotation_matrices.ipynb)
   — encode orientation changes.
5. [Matrix Kronecker product](01_linear_algebra/matrix_kronecker_product.ipynb)
   — compose an object with a larger spatial layout.
6. ARC tasks — infer which object, relation, and operator explain all
   demonstration pairs.

This path connects naturally to FeatureGraph:

```text
grid observations
    → bounded objects
    → spatial relations
    → candidate operators
    → composed output
```

## Notebook contract

When a notebook is revisited, add a short markdown header with:

- **Question** — what is the notebook trying to understand?
- **Input** — what representation enters the operation?
- **Operator** — what changes or composes the representation?
- **Output** — what representation is produced?
- **Invariant** — what remains unchanged?
- **Connection** — where does this operation appear elsewhere?
- **Open question** — what is not yet understood?

Exploration should remain in notebooks. Only operations that have a clear
contract and repeat across problems should move into reusable source code.

## Migration policy

- Notebook contents and outputs are preserved.
- Destination filenames use `snake_case`.
- The source repository is retained as provenance.
- Organization here expresses research relationships; it does not imply that
  every notebook is complete or correct.
