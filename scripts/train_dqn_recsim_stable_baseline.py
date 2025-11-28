import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
from gymnasium.envs.registration import register

from recgym.envs.rec import RECEnv

import matplotlib.pyplot as plt

# -------------------------------------------------------
# Register REC Environment
# -------------------------------------------------------
register(
    id="RECEnv-v0",
    entry_point="recgym.envs.rec:RECEnv",
    kwargs={"random_state": np.random.RandomState(42)},
)

# -------------------------------------------------------
# Plotting Helper
# -------------------------------------------------------
def plot_rewards(rewards, window=10, title="DQN Training Rewards"):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Episode Reward", alpha=0.6)

    if len(rewards) >= window:
        moving_avg = [np.mean(rewards[max(0, i - window + 1): i + 1])
                      for i in range(len(rewards))]
        plt.plot(moving_avg, label=f"{window}-Episode Moving Avg")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# DQN Model
# -------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# -------------------------------------------------------
# Replay Buffer
# -------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        s, a, r, ns, d = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(ns),
            np.array(d),
        )

    def __len__(self):
        return len(self.buffer)


# -------------------------------------------------------
# DQN Agent
# -------------------------------------------------------
class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.batch_size = 64
        self.update_target_steps = 100
        self.step_count = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state)
        return q.argmax().item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(
            self.device
        )
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_vals = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.MSELoss()(q_vals, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
def train_dqn(env_name, episodes, max_steps):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    rewards_history = []

    for ep in range(episodes):
        state, info = env.reset()
        total_reward = 0

        for t in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.buffer.push(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward

            agent.update()

            if terminated or truncated:
                break

        rewards_history.append(total_reward)
        print(
            f"[DQN] Episode {ep+1}/{episodes} | Reward={total_reward:.2f} | Epsilon={agent.epsilon:.3f}"
        )

    return rewards_history


# -------------------------------------------------------
# CLI + main entry
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="RECEnv-v0",
                        help="Environment ID ('RECEnv-v0' or 'LunarLander-v2')")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--plot", action="store_true", help="Show reward plot")

    args = parser.parse_args()

    rewards = train_dqn(
        env_name=args.env,
        episodes=args.episodes,
        max_steps=args.max_steps,
    )

    if args.plot:
        plot_rewards(rewards, window=10)
