# RL Agents Archive

This directory preserves superseded, incomplete, or duplicate experiments from
the original [`habibdraft/rl-agents`](https://github.com/habibdraft/rl-agents)
repository.

The files remain part of the research record. They are archived so that the
active directory can support one coherent tutorial without implying that every
historical implementation is current or runnable.

## Archived notebooks

| File | Reason retained |
|---|---|
| [`keras classifiers 2.ipynb`](notebooks/keras%20classifiers%202.ipynb) | Keras and tensor-shape experiments, including a useful recorded shape failure, but not a coherent RL lesson. |
| [`keras classifiers.ipynb`](notebooks/keras%20classifiers.ipynb) | Early TensorFlow DQN attempt; retained as an alternative implementation history. |
| [`visualizations.ipynb`](notebooks/visualizations.ipynb) | Earlier CartPole observation/visualization experiment superseded by `CartPole visualizations 1.ipynb`. |

## Archived modules

| File | Reason retained |
|---|---|
| [`agents.py`](legacy/agents.py) | Early tabular Q-learning/SARSA abstraction; contains shared-list initialization that aliases Q-table rows. |
| [`q-learning.py`](legacy/q-learning.py) | Taxi Q-learning experiment using the older Gym API and incomplete imports. |
| [`sarsa.py`](legacy/sarsa.py) | Taxi SARSA experiment using the older Gym API and incomplete episode handling. |
| [`dqn.py`](legacy/dqn.py) | Partial TensorFlow DQN and replay-buffer implementation with missing imports. |
| [`cp_agent.py`](legacy/cp_agent.py) | PyTorch DQN extraction that duplicates the later CartPole solution notebook. |

## Policy

- Archived files are not deleted or silently corrected.
- Their original contents are preserved.
- A historical artifact may return to active development only after its
  contract is made explicit and its behavior is covered by a focused example
  or test.
