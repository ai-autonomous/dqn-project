# Rainbow DQN (PyTorch) for LunarLander-v3

import gymnasium as gym
import numpy as np
import random
from collections import deque, namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------- Hyperparameters --------------------
ENV_NAME = "LunarLander-v3"
SEED = 42
GAMMA = 0.99
N_ATOMS = 51
V_MIN = -200.0
V_MAX = 200.0
N_STEPS = 3              # n-step returns
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
LR = 1e-4
UPDATE_TARGET_EVERY = 1000
TRAIN_START = 1000
MAX_FRAMES = 400_000
PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_FRAMES = 200_000

# -------------------- Utilities --------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# -------------------- Noisy Linear --------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# -------------------- Dueling C51 Network --------------------
class RainbowC51(nn.Module):
    def __init__(self, state_dim, action_dim, atoms=N_ATOMS):
        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)

        # value stream
        self.val_fc = NoisyLinear(256, 128)
        self.val_out = NoisyLinear(128, atoms)

        # advantage stream
        self.adv_fc = NoisyLinear(256, 128)
        self.adv_out = NoisyLinear(128, action_dim * atoms)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        v = F.relu(self.val_fc(x))
        v = self.val_out(v).view(-1, 1, self.atoms)  # (batch, 1, atoms)

        a = F.relu(self.adv_fc(x))
        a = self.adv_out(a).view(-1, self.action_dim, self.atoms)  # (batch, A, atoms)

        q_atoms = v + a - a.mean(1, keepdim=True)  # (batch, A, atoms)
        dist = F.softmax(q_atoms, dim=2)  # distribution over atoms
        dist = dist.clamp(min=1e-6)  # numerical stability
        return dist

    def reset_noise(self):
        self.val_fc.reset_noise(); self.val_out.reset_noise()
        self.adv_fc.reset_noise(); self.adv_out.reset_noise()

# -------------------- N-step + PER replay buffer (C51-friendly) --------------------
class PrioritizedReplayNstep:
    def __init__(self, capacity, n_step=N_STEPS, gamma=GAMMA, alpha=PRIORITY_ALPHA):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        # small queue for n-step accumulation
        self.nstep_queue = deque(maxlen=n_step)

        self.Transition = namedtuple('Transition', ['s', 'a', 'r', 'ns', 'd'])

    def _get_n_step_info(self):
        """Compute n-step cumulative reward."""
        reward, next_state, done = 0, None, None

        for idx, (_, _, r, ns, d) in enumerate(self.nstep_queue):
            reward += (self.gamma ** idx) * r
            next_state = ns
            done = d
            if d:   # terminate early if done occurs within n-step window
                break

        return reward, next_state, done

    def push(self, state, action, reward, next_state, done):
        """Stores both 1-step transitions and n-step transitions."""
        # Add to n-step queue
        self.nstep_queue.append((state, action, reward, next_state, done))

        # not enough for n-step yet
        if len(self.nstep_queue) < self.n_step:
            return

        # build n-step transition
        s, a, _, _, _ = self.nstep_queue[0]
        R, ns, d = self._get_n_step_info()

        transition = self.Transition(s, a, R, ns, d)

        # PER priority
        max_prio = self.priorities.max() if self.buffer else 1.0

        # store transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None

        prios = self.priorities[:len(self.buffer)]
        prios = np.maximum(prios, 1e-6)

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        samples = [self.buffer[i] for i in indices]
        batch = self.Transition(*zip(*samples))

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        states = torch.tensor(np.array(batch.s), dtype=torch.float32)
        actions = torch.tensor(batch.a, dtype=torch.long)
        rewards = torch.tensor(batch.r, dtype=torch.float32)
        next_states = torch.tensor(np.array(batch.ns), dtype=torch.float32)
        dones = torch.tensor(batch.d, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = p

    def __len__(self):
        return len(self.buffer)
# -------------------- Distributional projection (C51) --------------------
def projection_distribution(next_dist, rewards, dones, gamma):
    # next_dist: (batch, atoms)
    batch_size = rewards.size(0)
    proj_dist = torch.zeros((batch_size, N_ATOMS), dtype=torch.float32)
    delta_z = float(V_MAX - V_MIN) / (N_ATOMS - 1)
    support = torch.linspace(V_MIN, V_MAX, N_ATOMS)

    for i in range(batch_size):
        for j in range(N_ATOMS):
            Tz = rewards[i].item() + (gamma * support[j].item()) * (1.0 - dones[i].item())
            Tz = max(V_MIN, min(V_MAX, Tz))
            b = (Tz - V_MIN) / delta_z
            l = math.floor(b)
            u = math.ceil(b)
            if l == u:
                proj_dist[i, l] += next_dist[i, j]
            else:
                proj_dist[i, l] += next_dist[i, j] * (u - b)
                proj_dist[i, u] += next_dist[i, j] * (b - l)
    return proj_dist

# -------------------- Agent / Training loop --------------------
class RainbowAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        self.z = torch.linspace(V_MIN, V_MAX, N_ATOMS).to(device)
        self.delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)

        self.online = RainbowC51(state_dim, action_dim).to(device)
        self.target = RainbowC51(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=LR)

        self.replay = PrioritizedReplayNstep(BUFFER_SIZE)
        self.beta_start = PRIORITY_BETA_START
        self.beta_frames = PRIORITY_BETA_FRAMES
        self.frame_idx = 0

    def select_action(self, state):
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.online(state_v)  # (1, A, atoms)
            q_vals = torch.sum(dist * self.z, dim=2)  # (1, A)
            action = q_vals.argmax(1).item()
        return action

    def train_step(self):
        if len(self.replay) < BATCH_SIZE:
            return None
        self.frame_idx += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame_idx / float(self.beta_frames)))
        samples = self.replay.sample(BATCH_SIZE, beta)
        if samples is None:
            return None
        states, actions, rewards, next_states, dones, idxs, weights = samples
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        with torch.no_grad():
            # Double DQN: choose action with online net, evaluate with target
            next_dist = self.online(next_states)  # (B, A, atoms)
            next_q = torch.sum(next_dist * self.z, dim=2)  # (B, A)
            next_actions = next_q.argmax(1)  # (B,)
            next_dist_target = self.target(next_states)
            next_dist_target_a = next_dist_target[range(BATCH_SIZE), next_actions]  # (B, atoms)

            # Project distribution
            proj_dist = projection_distribution(next_dist_target_a, rewards, dones, GAMMA ** N_STEPS)
            proj_dist = proj_dist.to(self.device)

        dist = self.online(states)  # (B, A, atoms)
        dist_a = dist[range(BATCH_SIZE), actions]  # (B, atoms)

        # Loss: cross-entropy between proj_dist and dist_a
        loss = - (proj_dist * dist_a.log()).sum(1)
        loss = (weights * loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities
        with torch.no_grad():
            # compute TD error as KL divergence / or L1 between distributions' expected values
            expected = torch.sum(dist_a * self.z, dim=1)
            target_expected = torch.sum(proj_dist * self.z, dim=1)
            td_errors = (expected - target_expected).abs().cpu().numpy()
        self.replay.update_priorities(idxs, td_errors + 1e-6)

        # reset noisy nets
        self.online.reset_noise(); self.target.reset_noise()
        return loss.item()

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

# -------------------- Main training function --------------------

def train(episodes=800):
    env = gym.make(ENV_NAME, disable_env_checker=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = RainbowAgent(state_dim, action_dim, device)

    frame = 0
    losses = []
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done_flag = terminated or truncated
            agent.replay.push(state, action, reward, next_state, done_flag)
            state = next_state
            ep_reward += reward
            frame += 1

            if frame > TRAIN_START and frame % 1 == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

            if frame % UPDATE_TARGET_EVERY == 0:
                agent.update_target()

            if done_flag:
                break

        print(f"Episode {ep} Frame {frame} Reward {ep_reward}")

    env.close()
    return agent

if __name__ == '__main__':
    trained = train(episodes=400)
    torch.save(trained.online.state_dict(), 'rainbow_c51_lunar_v3.pth')
