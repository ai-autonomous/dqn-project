import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_ddqn(env, episodes=500, render_every=50):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=10000)
    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    update_target_every = 10
    rewards_per_episode = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if render_every and ep % render_every == 0:
                env.render()

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(policy_net(torch.FloatTensor(state))).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.BoolTensor(dones)

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_actions = torch.argmax(policy_net(next_states), dim=1)
                next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                targets = rewards + gamma * next_q_values * (~dones)

                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards_per_episode.append(total_reward)

        if ep % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(0.01, epsilon * 0.995)

        print(f"Episode {ep+1}/{episodes} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

    env.close()

    # Plotting the rewards
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DDQN Training Progress")
    plt.grid(True)
    plt.show()

# Example usage:
env = gym.make("CartPole-v1", render_mode="human")
train_ddqn(env)
