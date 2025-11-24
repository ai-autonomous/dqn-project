"""
Improved Double DQN for CartPole-v1 (Stable-Baselines3)
------------------------------------------------------
Includes:
  • Best-episode video recording (only when reward improves)
  • Two evaluation metrics:
        1) Official termination (FAIL/TIME_LIMIT)
        2) Stability heuristic (SOLVED/GOOD_RUN/UNSTABLE/FAIL)
  • Average loss plot + training reward progression
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

# ========= CLI Inputs =========
parser = argparse.ArgumentParser(description="Train Double DQN on CartPole-v1")
parser.add_argument("--total_steps", type=int, default=300_000)
parser.add_argument("--stage_size", type=int, default=50_000)
parser.add_argument("--lr", type=float, default=5e-4)
args = parser.parse_args()

# ========= Config =========
ENV_NAME = "CartPole-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models"
VIDEO_DIR = os.path.join(MODEL_DIR, "video")
MODEL_PATH = os.path.join(MODEL_DIR, "double_dqn_cartpole_v1.zip")
TB_LOG = "./tb_double_dqn_cartpole_v1"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

print(f"🚀 Training Double-DQN on {ENV_NAME} | Device: {DEVICE}")


# ========= LOSS LOGGER =========
class LossLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_losses = []
        self.current_losses = []

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

    def save(self):
        if not self.episode_losses:
            print("⚠️ No loss tracked.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.episode_losses, marker="o", alpha=0.8, color="red")
        plt.title("📉 Average Loss Per Episode (CartPole)")
        plt.xlabel("Episode")
        plt.ylabel("Avg Loss")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "loss_plot_cartpole.png"))
        print("💾 Saved loss plot!")


# ========= OFFICIAL TERMINATION =========
def official_termination(obs, terminated, truncated, steps, max_steps):
    if terminated:
        return "FAIL"
    if truncated and steps == max_steps:
        return "TIME_LIMIT"
    return "UNKNOWN"


# ========= STABILITY HEURISTIC =========
def quality_heuristic(obs, terminated, truncated, steps, max_steps):
    x, x_dot, theta, theta_dot = obs

    angle_ok = abs(theta) < 0.04
    ang_vel_ok = abs(theta_dot) < 0.15
    pos_ok = abs(x) < 0.7
    vel_ok = abs(x_dot) < 0.7

    stable = angle_ok and ang_vel_ok and pos_ok and vel_ok

    if truncated and steps == max_steps and stable:
        return "SOLVED"
    if terminated and steps >= 195:
        return "GOOD_RUN"
    if terminated:
        return "FAIL"
    return "UNSTABLE"


# ---------- ENV FACTORY ----------
def make_env(seed=0, record=False, video_name="best_cartpole"):
    # NOTE: must always use rgb_array if we might record later
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    if record:
        env = RecordVideo(
            env,
            VIDEO_DIR,
            name_prefix=video_name,
            episode_trigger=lambda episode_id: True,   # record every episode in this env
            disable_logger=True
        )

    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ---------- BEST VIDEO RECORDER ----------
best_reward = -np.inf

def record_best_video(model):
    global best_reward

    # Always create a fresh recording env (1 episode only)
    env = make_env(seed=777, record=True, video_name="best_cartpole")
    obs, _ = env.reset()
    total_r, steps = 0, 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total_r += reward
        steps += 1
        if term or trunc:
            break

    # VERY IMPORTANT flush & close properly
    try:
        env.close()
    except Exception:
        pass

    # Only keep if better
    if total_r > best_reward:
        best_reward = total_r
        print(f"🎥 BEST VIDEO SAVED! Reward={total_r:.2f}, Steps={steps}")
    else:
        # do not delete; simply next best will overwrite prefix names
        print(f"⚠️ Better score not achieved. Video skipped. Reward={total_r:.2f}")


# ========= TRAIN =========
def train_double_dqn(total_steps, stage_size, lr):
    env, eval_env = make_env(0), make_env(100)
    loss_logger = LossLogger()

    if os.path.exists(MODEL_PATH):
        print("📦 Loading existing model...")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        print("🆕 Creating new model...")
        model = DQN(
            "MlpPolicy", env,
            learning_rate=lr,
            buffer_size=50_000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            target_update_interval=250,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            verbose=1, seed=0,
            tensorboard_log=TB_LOG,
            device=DEVICE, policy_kwargs=dict(net_arch=[128, 128]),
        )

    stages = total_steps // stage_size
    rewards_track = []

    for s in range(stages):
        print(f"\n=== 🧠 Stage {s+1}/{stages} — {stage_size:,} steps ===")
        model.learn(stage_size, reset_num_timesteps=False, callback=loss_logger)
        model.save(MODEL_PATH)

        mean_r, std_r = evaluate_policy(model, eval_env, 10)
        rewards_track.append(mean_r)
        print(f"📈 Eval: {mean_r:.2f} ± {std_r:.2f}")

        record_best_video(model)  # 🎥 Save best only

    # Save loss plot
    loss_logger.save()

    # Reward plot
    plt.figure(figsize=(8, 4))
    plt.plot(rewards_track, marker="o")
    plt.title("📈 Mean Reward During Training (CartPole)")
    plt.xlabel("Stage"); plt.ylabel("Mean Reward")
    plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "reward_plot_cartpole.png"))

    env.close(); eval_env.close()
    return model


# ========= EVALUATION (TWO METRICS) =========
def evaluate(model, n=20):
    env = make_env(999)
    max_steps = env.env.spec.max_episode_steps

    official = {"FAIL": 0, "TIME_LIMIT": 0, "UNKNOWN": 0}
    quality = {"SOLVED": 0, "GOOD_RUN": 0, "UNSTABLE": 0, "FAIL": 0}

    for ep in range(n):
        obs,_ = env.reset()

        total_r, steps = 0, 0
        while True:
            action,_ = model.predict(obs, deterministic=True)
            obs,reward,term,trunc,_ = env.step(action)
            total_r += reward; steps += 1
            if term or trunc:
                official[official_termination(obs, term, trunc, steps, max_steps)] += 1
                quality[quality_heuristic(obs, term, trunc, steps, max_steps)] += 1
                break
    env.close()

    print("\n=== 🧪 OFFICIAL TERMINATION ===")
    for k,v in official.items(): print(f"{k:>12}: {v}")

    print("\n=== 🎯 QUALITY HEURISTIC ===")
    for k,v in quality.items(): print(f"{k:>12}: {v}")


# ========= MAIN =========
if __name__ == "__main__":
    model = train_double_dqn(args.total_steps, args.stage_size, args.lr)
    evaluate(model, 20)
    print("🎉 Done!")
