import gymnasium as gym
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from feature_graph.features import get_base_features, get_feature_graph, get_features_from_state
from feature_graph.cp_project_constants import get_states, axes, random_action, correct_action
from feature_graph.operators import start, end, between, operators
from policy_net.policy_net import PolicyNet

states, actions, episode_start, timeseries = get_states(correct_action)

base_features = {}

for feature, signal in axes.items():
    signal = timeseries[signal]
    base_features[f'{feature}_right'] = signal > 0
    base_features[f'{feature}_left'] = signal < 0

feature_graph = {}

for name, mask in base_features.items():
    feature_graph[name] = {
        'level': 0,
        'tensor': mask,
    }

feature_names = list(feature_graph.keys())

X = torch.stack([
    feature_graph[name]["tensor"].float()
    for name in feature_names
], dim=1)

y = actions.long()

policy = PolicyNet(X.shape[1])

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

for epoch in range(200):
    logits = policy(X)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 25 == 0:
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean()
        print(epoch, loss.item(), acc.item())

env = gym.make("CartPole-v1")

episode_rewards = []

for ep in range(20):
    state, info = env.reset()
    total_reward = 0

    for t in range(500):
        graph, x_vec = get_features_from_state(state, axes, feature_names)

        with torch.no_grad():
            logits = policy(x_vec)
            action = logits.argmax(dim=1).item()

        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

env.close()

print("Average reward:", sum(episode_rewards) / len(episode_rewards))
print("Rewards:", episode_rewards)