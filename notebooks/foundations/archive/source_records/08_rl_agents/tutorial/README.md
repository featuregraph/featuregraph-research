# Reinforcement Learning as Explicit State Transformation

This tutorial is designed around inspectable representations rather than an
agent abstraction introduced all at once.

The central progression is:

```text
state → action → transition → reward accumulation → value update → approximation
```

Each notebook should expose its intermediate tables so that the information
flow can be checked directly.

## Planned notebooks

### 1. `01_state_action_reward.ipynb`

Use a small random walk to define:

- state
- available action
- reward
- terminal state
- episode

**Contract:** given a state and action, return the next state, reward, and
termination flag.

### 2. `02_cartpole_transition_table.ipynb`

Run a deterministic or seeded random CartPole policy and construct one row per
environment transition.

| episode | step | x | velocity | angle | angular_velocity | action | reward | terminated |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|

**Contract:** convert raw Gymnasium interaction into an explicit transition
table.

### 3. `03_reward_as_accumulation.ipynb`

Represent episode return as an accumulation over transition rewards:

```math
G_t = r_t + \gamma G_{t+1}
```

Compare undiscounted cumulative reward with discounted return.

**FeatureGraph connection:** reward is an accumulated behavioral measurement
over a bounded episode object.

### 4. `04_value_update_by_hand.ipynb`

Calculate a single tabular Q update using visible columns:

| current_q | reward | next_max_q | target | error | updated_q |
|---:|---:|---:|---:|---:|---:|

```math
Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]
```

**Contract:** update one state-action estimate from one transition.

### 5. `05_tabular_agent.ipynb`

Apply the same update repeatedly in a small discrete environment. Separate:

- environment dynamics
- policy
- value table
- update operator
- evaluation

### 6. `06_dqn_as_function_approximation.ipynb`

Explain why continuous CartPole states make a literal Q-table inconvenient,
then replace the table lookup with:

```math
Q(s,a;\theta)
```

The neural network changes the representation of the value function; it does
not change the transition or Bellman contracts established earlier.

## Source material

The tutorial will derive ideas from these retained notebooks:

- [`random walk.ipynb`](../random%20walk.ipynb)
- [`sample state space.ipynb`](../sample%20state%20space.ipynb)
- [`CartPole visualizations 1.ipynb`](../CartPole%20visualizations%201.ipynb)
- [`cartpole_solution.ipynb`](../cartpole_solution.ipynb)

These are source records, not tutorial chapters. New tutorial notebooks should
be written deliberately rather than produced by mechanically cleaning old
notebooks.

## Notebook contract

Every tutorial notebook should state:

1. Question
2. Input representation
3. Operator
4. Output representation
5. Invariant
6. Validation check
7. Connection to the next notebook
