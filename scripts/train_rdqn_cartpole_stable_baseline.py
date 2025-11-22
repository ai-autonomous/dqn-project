import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean()

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.alpha = alpha

    def add(self, transition, error):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            self.priorities[i] = error + 1e-5

def train_rainbow(env, episodes=500):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = DuelingDQN(state_dim, action_dim)
    target_net = DuelingDQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = PrioritizedReplayBuffer(10000)
    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    update_target_every = 10

    for ep in range(episodes):
        #state = env.reset()
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(policy_net(torch.FloatTensor(state))).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            #next_state, reward, done, _ = env.step(action)
            td_error = abs(reward)
            buffer.add((state, action, reward, next_state, done), td_error)
            state = next_state
            total_reward += reward

            if len(buffer.buffer) >= batch_size:
                batch, indices, weights = buffer.sample(batch_size)
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

                loss = (q_values - targets.detach()).pow(2) * weights
                buffer.update_priorities(indices, loss.detach().numpy())
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if ep % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(0.01, epsilon * 0.995)
        print(f"Episode {ep}, Reward: {total_reward}")

env = gym.make("CartPole-v1")
train_rainbow(env)
