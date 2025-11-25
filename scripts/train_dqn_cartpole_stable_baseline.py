"""
DQN training for CartPole-v1 (Stable-Baselines3)
------------------------------------------------
FEATURES:
  • Heuristic stability classification:
        ➤ SOLVED (full episode 500 steps + stable)
        ➤ GOOD_RUN (≥450 steps)
        ➤ FAIL (fell early)
  • Saves BEST video ONLY when classified SOLVED
  • Tracks & plots average loss & training reward
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
parser = argparse.ArgumentParser(description="Train DQN on CartPole-v1")
parser.add_argument("--total_steps", type=int, default=300_000)
parser.add_argument("--stage_size", type=int, default=50_000)
parser.add_argument("--lr", type=float, default=5e-4)
args = parser.parse_args()

# ========= Config =========
ENV_NAME = "CartPole-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models"
VIDEO_DIR = os.path.join(MODEL_DIR, "video")
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_cartpole_v1.zip")
TB_LOG = "./tb_dqn_cartpole_v1"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

print(f"🚀 Training DQN on {ENV_NAME} | Device: {DEVICE}")

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
        if "episode" in info:  # episode ended
            if self.current_losses:
                self.episode_losses.append(np.mean(self.current_losses))
            self.current_losses = []
        return True

    def save_plot(self):
        if not self.episode_losses:
            print("⚠️ No loss tracked.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.episode_losses, marker="o", alpha=0.8, color="red")
        plt.title("📉 Average Loss Per Episode (CartPole)")
        plt.xlabel("Episode"); plt.ylabel("Avg Loss")
        plt.grid(); plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "loss_plot_cartpole.png"))
        print("💾 Saved loss plot!")


# ========= STABILITY HEURISTIC =========
def classify_outcome(obs, terminated, truncated, steps, max_steps):
    """CartPole-v1 stability classification (500 step limit)"""
    x, x_dot, theta, theta_dot = obs

    # stable posture
    angle_ok   = abs(theta) < 0.04        # < 2.3°
    ang_vel_ok = abs(theta_dot) < 0.15
    pos_ok     = abs(x) < 0.7
    vel_ok     = abs(x_dot) < 0.7
    stable = angle_ok and ang_vel_ok and pos_ok and vel_ok

    if truncated and steps == max_steps and stable:
        return "SOLVED"      # best possible

    if terminated:
        return "FAIL"        # fell early

    if truncated and steps == max_steps:
        return "TIME_LIMIT"  # survived but unstable

    if steps >= 450 and stable:  # strong run for v1
        return "GOOD_RUN"

    return "UNSTABLE"


# ---------- ENV FACTORY ----------
def make_env(seed=0, record=False, fname="best_cartpole"):
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    if record:
        env = RecordVideo(
            env, VIDEO_DIR, name_prefix=fname,
            episode_trigger=lambda _: True, disable_logger=True
        )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ---------- VIDEO SAVER ----------
def save_best_video(model):
    """Record only if the episode is SOLVED"""
    env = make_env(seed=777, record=True, fname="best_cartpole")
    obs,_ = env.reset()
    total_r, steps = 0,0

    while True:
        action,_ = model.predict(obs, deterministic=True)
        obs,rew,term,trunc,_ = env.step(action)
        total_r += rew; steps += 1
        if term or trunc:
            outcome = classify_outcome(obs, term, trunc, steps, env.env.spec.max_episode_steps)
            break

    env.close()

    if outcome == "SOLVED":
        print(f"🎥 VIDEO SAVED (SOLVED episode: {steps} steps, reward={total_r:.2f})")
    else:
        print(f"⚠️ Not SOLVED ({outcome}), video discarded")


# ========= TRAIN =========
def train_dqn(total_steps, stage_size, lr):
    env, eval_env = make_env(0), make_env(100)
    loss_logger = LossLogger()

    if os.path.exists(MODEL_PATH):
        print("📦 Loading existing model...")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        print("🆕 Creating new model...")
        model = DQN(
            "MlpPolicy", env,
            learning_rate=lr, buffer_size=50_000, batch_size=128,
            tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1,
            target_update_interval=500, exploration_fraction=0.4,
            exploration_final_eps=0.05, verbose=1, seed=0,
            tensorboard_log=TB_LOG,
            device=DEVICE, policy_kwargs=dict(net_arch=[256,256]),
        )

    rewards_track = []
    stages = total_steps // stage_size

    for s in range(stages):
        print(f"\n=== 🧠 Stage {s+1}/{stages} — {stage_size:,} steps ===")
        model.learn(stage_size, reset_num_timesteps=False, callback=loss_logger)
        model.save(MODEL_PATH)

        mean_r,_ = evaluate_policy(model, eval_env, 10)
        rewards_track.append(mean_r)
        print(f"📈 Eval Mean Reward: {mean_r:.2f}")

        save_best_video(model)  # 🎥 Only SOLVED runs saved

    loss_logger.save_plot()

    plt.figure(figsize=(8,4))
    plt.plot(rewards_track, marker="o")
    plt.title("📈 Mean Reward During Training (CartPole)")
    plt.xlabel("Stage"); plt.ylabel("Mean Reward")
    plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "reward_plot_cartpole.png"))

    env.close(); eval_env.close()
    return model


# ========= MAIN =========
if __name__ == "__main__":
    model = train_dqn(args.total_steps, args.stage_size, args.lr)
    print(f"🎉 Finished! Videos (if any) saved in {VIDEO_DIR}")
