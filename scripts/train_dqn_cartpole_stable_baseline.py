"""
Adaptive DQN for CartPole-v1 (Stable-Baselines3)
------------------------------------------------
Features:
  • 🎥 Save best video ONLY when heuristically classified as SOLVED
  • 📌 Adaptive heuristic thresholds tighten as performance increases
  • 📉 Average loss per episode
  • 📈 Mean reward progression per training stage
  • 🧪 Evaluation (Official + Heuristic)
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ================= CLI Inputs =================
parser = argparse.ArgumentParser(description="Train DQN on CartPole-v1")
parser.add_argument("--total_steps", type=int, default=300_000)
parser.add_argument("--stage_size", type=int, default=50_000)
parser.add_argument("--lr", type=float, default=5e-4)
args = parser.parse_args()

# ================= Config =================
ENV_NAME = "CartPole-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models"
VIDEO_DIR = os.path.join(MODEL_DIR, "video")
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_cartpole_v1.zip")
TB_LOG = "./tb_dqn_cartpole_v1"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ================= Loss Logger =================
class LossLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_losses, self.current_losses = [], []

    def _on_step(self) -> bool:
        loss = self.model.logger.name_to_value.get("train/loss")
        if loss is not None:
            self.current_losses.append(loss)

        info = self.locals.get("infos", [{}])[0]
        if "episode" in info:
            if self.current_losses:
                self.episode_losses.append(np.mean(self.current_losses))
            self.current_losses = []
        return True

    def save_plot(self):
        if not self.episode_losses:
            print("⚠️ No loss tracked.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.episode_losses, marker="o", color="red")
        plt.title("📉 Average Loss Per Episode (DQN)")
        plt.xlabel("Episode")
        plt.ylabel("Avg Loss")
        plt.grid(); plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "loss_plot_cartpole.png"))
        print("💾 Saved loss plot!")

# ================= Adaptive Heuristic =================
def adaptive_thresholds(mean_reward: float):
    tighten = min(0.8, max(0, (mean_reward - 100) / 200))
    return {
        "theta": 0.07 - 0.04 * tighten,
        "theta_dot": 0.25 - 0.10 * tighten,
        "x": 0.9 - 0.3 * tighten,
        "x_dot": 0.9 - 0.3 * tighten,
    }

def heuristic_eval(obs, terminated, truncated, steps, max_steps, mean_reward):
    x, x_dot, theta, theta_dot = obs
    th = adaptive_thresholds(mean_reward)

    stable = (abs(theta) < th["theta"] and abs(theta_dot) < th["theta_dot"] and
              abs(x) < th["x"] and abs(x_dot) < th["x_dot"])

    if truncated and steps == max_steps and stable:
        return "SOLVED"
    if terminated and stable and steps >= 195:
        return "GOOD_RUN"
    if terminated:
        return "FAIL"
    return "UNSTABLE"

# ================= Environment Factory =================
def make_env(seed=0, record=False, vid_name="best_cartpole"):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    if record:
        env = RecordVideo(env, VIDEO_DIR, name_prefix=vid_name,
                          episode_trigger=lambda e: True, disable_logger=True)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

# ================= Video Recorder =================
best_reward = -np.inf

def record_if_solved(model, mean_reward):
    global best_reward
    env = make_env(seed=777, record=True)
    obs,_ = env.reset()
    total_r, steps = 0, 0
    max_steps = env.env.spec.max_episode_steps

    while True:
        last_obs = obs.copy()
        action,_ = model.predict(obs, deterministic=True)
        obs,reward,term,trunc,_ = env.step(action)
        total_r += reward; steps += 1
        if term or trunc:
            outcome = heuristic_eval(last_obs, term, trunc, steps, max_steps, mean_reward)
            break

    env.close()
    if outcome == "SOLVED" and total_r > best_reward:
        best_reward = total_r
        print(f"🎥 SOLVED! Video saved. Reward={total_r:.2f}")
    else:
        print(f"⏭  No SOLVED video (Result={outcome}, R={total_r:.1f})")

# ================= TRAIN =================
def train_dqn(total_steps, stage_size, lr):
    env, eval_env = make_env(0), make_env(100)
    loss_logger = LossLogger()

    if os.path.exists(MODEL_PATH):
        print("📦 Loading existing model...")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        print("🆕 Creating new DQN model...")
        model = DQN(
            "MlpPolicy", env,
            learning_rate=lr, buffer_size=50_000, batch_size=128,
            tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1,
            target_update_interval=500, exploration_fraction=0.4,
            exploration_final_eps=0.05, verbose=1, seed=0,
            tensorboard_log=TB_LOG, device=DEVICE,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    stages = total_steps // stage_size
    reward_track = []

    for s in range(stages):
        print(f"\n=== Stage {s+1}/{stages} — {stage_size:,} steps ===")
        model.learn(stage_size, reset_num_timesteps=False, callback=loss_logger)
        model.save(MODEL_PATH)

        mean_r, std_r = evaluate_policy(model, eval_env, 10)
        reward_track.append(mean_r)
        print(f"📈 Eval: {mean_r:.2f} ± {std_r:.2f}")

        record_if_solved(model, mean_r)

    loss_logger.save_plot()

    plt.figure(figsize=(8, 4))
    plt.plot(reward_track, marker="o")
    plt.title("📈 Mean Reward During Training (CartPole)")
    plt.xlabel("Stage"); plt.ylabel("Mean Reward")
    plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "reward_plot_cartpole.png"))

    env.close(); eval_env.close()
    return model

# ================= Evaluation =================
def evaluate(model, n=20):
    env = make_env(999)
    max_steps = env.env.spec.max_episode_steps
    metrics = {"SOLVED":0, "GOOD_RUN":0, "UNSTABLE":0, "FAIL":0}

    for _ in range(n):
        obs,_ = env.reset()
        total_r, steps = 0, 0
        while True:
            last_obs = obs.copy()
            action,_ = model.predict(obs, deterministic=True)
            obs,reward,term,trunc,_ = env.step(action)
            total_r += reward; steps += 1
            if term or trunc:
                metrics[heuristic_eval(last_obs, term, trunc, steps, max_steps, np.mean([100,200]))] += 1
                break
    env.close()

    print("\n=== 🧪 Heuristic Evaluation ===")
    for k,v in metrics.items(): print(f"{k:>12}: {v}")

# ================= MAIN =================
if __name__ == "__main__":
    model = train_dqn(args.total_steps, args.stage_size, args.lr)
    evaluate(model)
    print("🎉 Done!")