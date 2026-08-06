import gymnasium as gym
import pandas as pd
import torch
import random

def random_action(state):
    return random.choice([0, 1])
def correct_action(state):
    x, v, theta, omega = state
    score = theta + 0.5 * omega + 0.05 * x + 0.05 * v
    return 1 if score > 0 else 0

def get_states(op):
    env = gym.make("CartPole-v1")

    rows = []
    state, info = env.reset()
    episode = 0
    t_in_episode = 0

    while len(rows) < 20_000:
        action = op(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        x, v, theta, omega = state

        rows.append({
            "row": len(rows),
            "episode": episode,
            "t": t_in_episode,
            "x": x,
            "v": v,
            "theta": theta,
            "omega": omega,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "episode_start": t_in_episode == 0,
        })

        if terminated or truncated:
            state, info = env.reset()
            episode += 1
            t_in_episode = 0
        else:
            state = next_state
            t_in_episode += 1

    env.close()

    df = pd.DataFrame(rows)

    states = torch.tensor(
        df[["x", "v", "theta", "omega"]].values,
        dtype=torch.float32
    )

    actions = torch.tensor(df["action"].values, dtype=torch.long)

    episode_start = torch.tensor(
        df["episode_start"].values,
        dtype=torch.bool
    )

    timeseries = {
        'x': states[:, 0],
        'v': states[:, 1],
        'theta': states[:, 2],
        'omega': states[:, 3],
        'episode_start': episode_start
    }

    return states, actions, episode_start, timeseries

axes = {
    'cart': 'x',
    'cart_moving': 'v',
    'pole': 'theta',
    'pole_rotating': 'omega'
}