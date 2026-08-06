# Reinforcement Learning Foundations

This directory preserves the original RL-agent research while providing a
clean path toward a representation-first tutorial.

## Start here

Read [`tutorial/README.md`](tutorial/README.md) for the planned six-notebook
sequence:

```text
state → action → transition → reward accumulation → value update → approximation
```

## Active source material

| Artifact | Tutorial role |
|---|---|
| [`random walk.ipynb`](random%20walk.ipynb) | Small discrete states and manual value updates |
| [`sample state space.ipynb`](sample%20state%20space.ipynb) | Explicit state/action/reward construction |
| [`CartPole visualizations 1.ipynb`](CartPole%20visualizations%201.ipynb) | CartPole observations and transition-table exploration |
| [`cartpole_solution.ipynb`](cartpole_solution.ipynb) | Advanced PyTorch DQN reference |

## Archive

Superseded, duplicate, or incomplete experiments are documented in
[`archive/README.md`](archive/README.md). Archiving preserves the research
history while keeping the active workflow legible.

## Principle

The tutorial begins with inspectable state and transition tables. Agent classes
and neural networks are introduced only after their contracts are visible in
the data.
