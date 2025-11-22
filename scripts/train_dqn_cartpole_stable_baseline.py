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

def train_dqn(env, episodes=500, save_path="dqn_cartpole.pth"):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=10000)
    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = truncated = False
        while not (terminated or truncated):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(model(torch.FloatTensor(state))).item()

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

                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = model(next_states).max(1)[0]
                targets = rewards + gamma * next_q_values * (~dones)

                loss = criterion(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(0.01, epsilon * 0.995)
        rewards_history.append(total_reward)
        print(f"Episode {ep}, Reward: {total_reward}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Plot training rewards
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Episodes")
    plt.grid()
    plt.show()

    return model

def evaluate_dqn(env, model, episodes=10):
    print("\nEvaluating trained model...")
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = truncated = False
        while not (terminated or truncated):
            with torch.no_grad():
                action = torch.argmax(model(torch.FloatTensor(state))).item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        print(f"Evaluation Episode {ep}, Reward: {total_reward}")

# Train and evaluate
env = gym.make("CartPole-v1")
trained_model = train_dqn(env)
evaluate_dqn(env, trained_model)
