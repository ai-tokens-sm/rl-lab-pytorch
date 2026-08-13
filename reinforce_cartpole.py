"""
Minimal REINFORCE (policy gradient) agent for CartPole-v1.
Trains a small neural network policy using PyTorch and Gymnasium.
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)  # returns action logits


def select_action(policy, state):
    state = torch.as_tensor(state, dtype=torch.float32)
    logits = policy(state)
    dist = Categorical(logits=logits)
    action = dist.sample()
    return action.item(), dist.log_prob(action)


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize
    return returns


def train(num_episodes=500, gamma=0.99, lr=1e-2):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = Policy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    running_reward = 0.0

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        log_probs, rewards = [], []
        done = False

        while not done:
            action, log_prob = select_action(policy, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)

        returns = compute_returns(rewards, gamma)
        loss = -torch.stack([lp * G for lp, G in zip(log_probs, returns)]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        running_reward = 0.05 * episode_reward + 0.95 * running_reward

        if episode % 20 == 0:
            print(f"Episode {episode:4d} | reward: {episode_reward:6.1f} | running avg: {running_reward:6.1f}")

        if running_reward > 475:
            print(f"Solved at episode {episode}! Running reward: {running_reward:.1f}")
            break

    env.close()
    return policy


if __name__ == "__main__":
    train()