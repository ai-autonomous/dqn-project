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
import argparse

# -----------------------------
# Register environment
# -----------------------------
register(
    id="RECEnv-v0",
    entry_point="recgym.envs.rec:RECEnv",
    kwargs={"random_state": np.random.RandomState(42)},
)

# -----------------------------
# Helper — Reward Plot
# -----------------------------
def plot_rewards(rewards, window=10):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward', alpha=0.6)

    if len(rewards) >= window:
        moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        plt.plot(moving_avg, label=f'{window}-Episode Moving Avg', color='red')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DDQN Training Rewards")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------
# Q-Network
# -----------------------------
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

# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# DDQN Agent
# -----------------------------
class DDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        buffer_size,
        batch_size,
        update_target_steps,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_steps = update_target_steps

        # Epsilon
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(capacity=buffer_size)

        self.step_count = 0

    # ε-greedy
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.policy_net(state_tensor)
        return qvals.argmax().item()

    # Double-DQN update
    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # DDQN:
        # 1. Action chosen by policy net
        next_policy_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)

        # 2. Action evaluated by target net
        with torch.no_grad():
            next_q = self.target_net(next_states).gather(1, next_policy_actions)
            q_target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Target sync
        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# -----------------------------
# Training Loop
# -----------------------------
def train_ddqn(env_name, episodes, max_steps, args):
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        update_target_steps=args.update_target_steps,
    )

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
        print(f"[DDQN] Episode {ep+1}/{episodes}  Reward={total_reward:.2f}  Epsilon={agent.epsilon:.3f}")

    return rewards_history

# -----------------------------
# Main + CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--plot", type=str, default="true", help="true/false")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)

    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_target_steps", type=int, default=100)

    args = parser.parse_args()

    rewards = train_ddqn(
        env_name="RECEnv-v0",
        episodes=args.episodes,
        max_steps=args.max_steps,
        args=args
    )

    if args.plot.lower() == "true":
        plot_rewards(rewards, window=10)
