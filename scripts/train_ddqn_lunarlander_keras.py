# DDQN LunarLander Implementation - Keras
import os
import argparse
import random
import time
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from gymnasium.wrappers import RecordVideo

# ----------------- CLI Args -----------------
parser = argparse.ArgumentParser(description="Train DDQN on LunarLander-v3")
parser.add_argument("--total_steps", type=int, default=400_000)
parser.add_argument("--stage_size", type=int, default=50_000)
parser.add_argument("--lr", type=float, default=5e-4)
args = parser.parse_args()

# ----------------- Hyperparameters -----------------
ENV_NAME = "LunarLander-v3"
SEED = 1234
BUFFER_SIZE = 200_000  
BATCH_SIZE = 64
GAMMA = 0.99
LR = 5e-4              
TRAIN_EVERY = 1        
TRAIN_START = 5_000      
MAX_FRAMES = 400_000
MAX_EPISODES = 400
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_FRAMES = 120_000  
TARGET_UPDATE = 1500    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Track Metrics
ALL_REWARDS = []
ALL_LOSSES = []

MODEL_DIR = Path("models")
VIDEO_DIR = os.path.join(MODEL_DIR, "best_video")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
MODEL_FILE = MODEL_DIR / "ddqn_lunarlander_keras.pth"
LOG_FILE = Path("ddqn_lunarlander_keras_results.txt")
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_lunarlander_v3.zip")
LOSS_CSV = os.path.join(MODEL_DIR, "loss_log.csv")
REWARD_CSV = os.path.join(MODEL_DIR, "reward_log.csv")

LOSS_CSV = os.path.join(MODEL_DIR, "loss_log.csv")

# ----------------- Utilities -----------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ---------- TERMINATION REASON ----------
def termination_reason(env, obs):
    """Detect landing outcome in LunarLander-v3"""
    um = env.unwrapped
    try:
        x_pos = float(obs[0])
        if getattr(um, "game_over", False):
            return "CRASH"
        elif abs(x_pos) >= 2.5:
            return "OUT_OF_BOUNDS"
        elif not bool(um.lander.awake):
            left, right = int(obs[6]), int(obs[7])
            return "LANDED_OK" if left == 1 and right == 1 else "ASLEEP"
        else:
            return "UNKNOWN"
    except:
        return "DONE"

# ---------- ENV FACTORY ----------
def make_env(seed=0, record=False, tag=""):
    env = gym.make(ENV_NAME, render_mode="rgb_array" if record else None)
    if record:
        env = RecordVideo(env, VIDEO_DIR, name_prefix=f"best_landing_{tag}")
    #env = Monitor(env)
    env.reset(seed=seed)
    return env

# ----------------- Replay Buffer -----------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        from collections import deque

        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ----------------- Q Network -----------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


# ----------------- Agent -----------------
class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = args.lr, buffer_size: int = BUFFER_SIZE):
        self.action_dim = action_dim
        self.online = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.steps = 0

    def act(self, state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.action_dim)
        state_v = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            qvals = self.online(state_v)
            return int(qvals.argmax(1))
    def act_greedy(self, obs):
        # evaluation-time behavior (deterministic)
        state_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        q = self.online(state_t).detach().cpu().numpy()
        return int(np.argmax(q[0]).item())
    def train_step(self):
        if len(self.replay) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)
        states_v = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        next_states_v = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
        actions_v = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        dones_v = torch.tensor(dones.astype(np.uint8), dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q_vals = self.online(states_v).gather(1, actions_v)

        with torch.no_grad():
            next_q_online = self.online(next_states_v)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states_v).gather(1, next_actions)
            target_q = rewards_v + GAMMA * (1.0 - dones_v) * next_q_target

        loss = F.smooth_l1_loss(q_vals, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: Path):
        torch.save(self.online.state_dict(), str(path))

    def load(self, path: Path):
        self.online.load_state_dict(torch.load(str(path), map_location=DEVICE))
        self.target.load_state_dict(self.online.state_dict())


# ----------------- Main Training -----------------
def train(total_steps, stage_size, lr):
    set_seed(SEED)
    env = gym.make(ENV_NAME)

    # gymnasium vs gym compatibility for reset
    try:
        obs = env.reset(seed=SEED)
    except TypeError:
        obs = env.reset()

    # determine state/action sizes
    state_example = env.observation_space.sample()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DDQNAgent(state_dim, action_dim)
    episodes = total_steps/1000
    LOG_FILE.write_text("")
    episode = 0
    frame = 0
    eps = EPS_START
    best_avg = -1e9
    start_time = time.time()

    while episode < episodes and frame < total_steps:
        reset_ret = env.reset()
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        ep_reward = 0.0
        done = False

        while not done and frame < total_steps:
            frame += 1
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * frame / EPS_DECAY_FRAMES) * (frame / EPS_DECAY_FRAMES)

            action = agent.act(state, eps)
            step_ret = env.step(action)

            # gymnasium returns (obs, reward, terminated, truncated, info)
            if len(step_ret) == 5:
                next_state, reward, terminated, truncated, info = step_ret
                done_flag = bool(terminated or truncated)
            else:
                next_state, reward, done_flag, info = step_ret

            # improved reward shaping
            try:
                # encourage center
                reward += 1.0 * (1.0 - min(1.0, abs(next_state[0])))
                # maintain altitude
                reward += 0.5 * (1.0 - min(1.0, abs(next_state[1])))
                # stabilize angle
                reward += 1.0 * (1.0 - min(1.0, abs(next_state[4])))
                # penalize angular velocity
                reward -= 0.1 * abs(next_state[5])
                # leg contact bonus
                if len(next_state) >= 8:
                    reward += 5.0 * float(next_state[6])
                    reward += 5.0 * float(next_state[7])
            except Exception:
                pass

            agent.replay.push(state, action, reward, next_state, done_flag)
            state = next_state
            ep_reward += reward
 
            if frame > TRAIN_START and frame % TRAIN_EVERY == 0:
                loss = agent.train_step()
                if loss is not None:
                    ALL_LOSSES.append(loss)

            if frame % TARGET_UPDATE == 0:
                agent.update_target()

            if done_flag:
                # terminal landing bonus
                try:
                    vx, vy = float(next_state[2]), float(next_state[3])
                    ang = float(next_state[4])
                    landing_bonus = 50.0 - 20.0 * abs(vy) - 10.0 * abs(ang)
                    landing_bonus = max(0.0, landing_bonus)
                    ep_reward += landing_bonus
                except Exception:
                    pass
                break

        ALL_REWARDS.append(ep_reward)
        avg100 = float(np.mean(ALL_REWARDS[-100:])) if len(ALL_REWARDS) >= 1 else ep_reward
        line = f"Episode {episode:4d} Frame {frame:6d} Reward {ep_reward:8.2f} AvgRecent {avg100:8.2f}"

        LOG_FILE.write_text(LOG_FILE.read_text() + line)
        print(line.strip())

        if avg100 > best_avg and avg100 >= 200.0:
            best_avg = avg100
            agent.save(MODEL_FILE)
            print(f"New best avg100 {best_avg:.2f} — model saved: {MODEL_FILE}")
            print(f"🎥 NEW BEST AVERAGE REWARD {best_avg:.2f} → Recording episode")
            record_best_video(agent)
        if (episode*1000) % stage_size == 0:
            agent.save(MODEL_FILE)
            print(f"New model saved at step {episode*1000}: {MODEL_FILE}")
        episode += 1

    env.close()
    elapsed = time.time() - start_time
    print(f"Training finished. Frames: {frame}, Episodes: {episode}, Time: {elapsed/60:.2f} min")
    agent.save(MODEL_FILE)
    print(f"Saved final model to {MODEL_FILE}")

    # ---------- Save Model ----------
    os.makedirs("Lunarlander_ddqnkeras_outputs", exist_ok=True)
    torch.save(agent.online.state_dict(), "Lunarlander_ddqnkeras_outputs/Lunarlander_ddqnkeras.pth")

    # ---------- Save Graphs ----------
    plt.figure(figsize=(8, 4))
    plt.plot(ALL_REWARDS, marker="o", alpha=0.7, color="blue")
    plt.title("📈 Reward Per Episode - DDQN LunarLander (Keras)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "reward_plot.png"))

    if ALL_LOSSES:
        plt.figure(figsize=(8, 4))
        plt.plot(ALL_LOSSES, marker="o", alpha=0.7, color="red")
        plt.title("📉 Loss Curve - DDQN LunarLander (Keras)")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "loss_plot.png"))

    print("📊 Saved loss & reward plots under ddqn_outputs")
    return agent, ALL_REWARDS, ALL_LOSSES

# ---------- RECORD BEST VIDEO ----------
def record_best_video(model):
    env = make_env(1234, record=True, tag="best")
    obs, _ = env.reset()

    while True:
        action = model.act_greedy(obs)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    env.close()

# ----------------- EVALUATION (PRINT SUMMARY) -----------------
def evaluate_and_report(agent: DDQNAgent, n_eps: int = 20):
    env = gym.make(ENV_NAME)
    outcomes = {"LANDED_OK": 0, "CRASH": 0, "OUT_OF_BOUNDS": 0, "ASLEEP": 0, "UNKNOWN": 0, "DONE": 0}
    rewards = []

    for ep in range(n_eps):
        reset_ret = env.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        ep_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                state_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_vals = agent.online(state_v)
                action = int(q_vals.argmax(1).item())

            step_ret = env.step(action)
            if len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_ret

            ep_reward += reward

            if done:
                reason = termination_reason(env, obs)
                outcomes[reason] = outcomes.get(reason, 0) + 1
                rewards.append(ep_reward)
                break
    env.close()
    
    print("=== Evaluation Summary ===")
    for k, v in outcomes.items():
        print(f"{k:>15}: {v}")
    print(f"Mean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

    # simple bar plot
    labels, counts = list(outcomes.keys()), list(outcomes.values())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts)
    plt.title("LunarLander Episode Outcomes")
    plt.tight_layout()
    plt.savefig(MODEL_DIR, "evaluation_summary.png")


# ----------------- Run -----------------
if __name__ == '__main__':
    print(f"🚀 Training on {ENV_NAME} | Steps={args.total_steps:,} | LR={args.lr}")
    agent, rewards, losses = train(args.total_steps, args.stage_size, args.lr)

    # load the saved/best model for evaluation if exists
    if MODEL_FILE.exists():
        agent.load(MODEL_FILE)

    evaluate_and_report(agent, n_eps=20)
    print("🎉 Done!")
    print("🎉 Done! Video saved in:", VIDEO_DIR)


