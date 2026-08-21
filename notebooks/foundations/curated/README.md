# Maintained Foundations

The maintained sequence has three tracks:

1. [Array and spatial representation](01_array_and_spatial_representation.md)
2. [Temporal and behavioral representation](02_temporal_and_behavioral_representation.md)
3. [Operators, provenance, and instructions](03_operators_provenance_and_instructions.md)

Each guide uses the same contract:

- **Question**: what must become explicit?
- **Input**: what representation enters?
- **Operator**: what deterministic operation is applied?
- **Output**: what representation is produced?
- **Invariant**: what must remain true?
- **Validation**: how is correctness checked?
- **Connection**: where does this representation enter FeatureGraph?

The guides explain the architecture. Tested implementations live in
`src/featuregraph/`; historical derivations live in `../archive/`.
