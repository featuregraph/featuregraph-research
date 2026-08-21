# Temporal and Behavioral Representation

## Question

How can a sample sequence be transformed into explicit states, boundaries,
events, and bounded behavioral objects?

## Contract

- **Input:** an ordered observation or Boolean state sequence.
- **Operator:** derive primitive states, detect entry and exit transitions, and
  materialize the intervals between matched boundaries.
- **Output:** sample-aligned state and event arrays plus bounded object records.
- **Invariant:** transition arrays preserve sample alignment and object
  boundaries follow the declared state contract.
- **Validation:** the state reconstructed from the boundary events matches the
  declared intervals.

## Representation lifecycle

```text
observations
    → primitive states
    → transition events
    → bounded behavioral objects
    → object-relative measurements
    → summaries and relations
```

A state says what is true at a sample. An event says where that truth changes.
An object binds a beginning, persistence interval, and ending into an identity.
A measurement describes that object rather than an isolated sample.

## NumPy transition operators

```python
from featuregraph.operators import between_masks, enter_mask, exit_mask

enter = enter_mask(state)
exit = exit_mask(state)
active = between_masks(enter, exit)
```

These are the reusable core hidden inside several archived Markov, RL, and
FeatureGraph-origin experiments. They do not introduce episodes, agents, or
domain thresholds. Those are contextual layers added after the basic temporal
contract is explicit.

## Accumulation

Accumulation is a measurement over a bounded interval, not a replacement for
state or identity. Rewards accumulated over an episode, material accumulated
during a process phase, and displacement accumulated during motion share the
same structural question:

```text
which contribution belongs to which bounded object?
```

This is why the historical RL progression remains conceptually useful even
though the old agent implementations are archived.

## Connection

Oscillation, accumulation, BIDMC respiration, TEP process behavior, and future
transition-based objects should all preserve the distinction between observed
quantity, primitive state, transition event, object identity, intrinsic
measurement, and external context.
