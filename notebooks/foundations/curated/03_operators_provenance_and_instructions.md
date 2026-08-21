# Operators, Provenance, and Instructions

## Question

How can FeatureGraph represent not only a derived array, but also the named
operation and contract that produced it?

## Contract

- **Input:** a representation and a named operator.
- **Operator:** apply a deterministic function while checking its declared
  invariants.
- **Output:** a derived representation with enough metadata to reproduce its
  construction.
- **Invariant:** operator identity is stable across datasets; numeric column
  positions are temporary execution details.
- **Validation:** the same named instruction applied to compatible inputs
  produces representations satisfying the same contract.

## Typed operator records

The FeatureGraph-origin prototype stored a tensor beside an operator name,
parents, and graph level. EventQL separately explored syntax trees, type
inference, compilation, and deterministic evaluation. The maintained
extraction is deliberately smaller:

```python
import numpy as np

from featuregraph.operators import OperatorRecord

operator = OperatorRecord(
    name="flip_horizontal",
    function=np.fliplr,
    input_kind="grid",
    output_kind="grid",
    preserves_shape=True,
)

result = operator.apply(grid)
```

This record makes operator identity and invariants explicit without committing
FeatureGraph to a textual DSL.

## Instruction layouts

An instruction layout should contain stable operator names:

```text
[[copy, flip_horizontal, flip_vertical],
 [flip_vertical, copy, flip_horizontal]]
```

During execution, names may be mapped to columns in a candidate-value matrix.
That numeric layout is local and disposable. The named layout is the learned,
transferable representation. Column `1` must not silently mean different
transformations for inputs whose valid candidate lists differ.

## Provenance direction

A future derived-representation record can extend this contract with parent
identifiers, parameters, shapes, validation evidence, implementation version,
study provenance, and a deterministic content hash.

The archived EventQL grammar and AST remain useful references for a future
expression intermediate representation. They are not part of the stable API
because the current priority is explicit operator contracts, not syntax.

The same structure applies to ARC instructions, state-contract compilation,
transition construction, reproducible studies, and auditable queries. The
durable object is the result plus a stable account of what produced it.
