"""
DQN training with loss & reward tracking + evaluation summary plots
DO NOT REMOVE: Evaluation summary print section
"""

import os
import argparse
import numpy as np
import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt
import torch
import warnings

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ---------- CLI Args ----------
parser = argparse.ArgumentParser(description="Train DQN on LunarLander-v3")
parser.add_argument("--total_steps", type=int, default=500_000)
parser.add_argument("--stage_size", type=int, default=100_000)
parser.add_argument("--lr", type=float, default=5e-4)
args = parser.parse_args()

ENV_NAME = "LunarLander-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models"
VIDEO_DIR = os.path.join(MODEL_DIR, "best_video")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_lunarlander_v3.zip")

LOSS_CSV = os.path.join(MODEL_DIR, "loss_log.csv")
REWARD_CSV = os.path.join(MODEL_DIR, "reward_log.csv")


# ---------- LOSS LOGGER ----------
class LossLogger(BaseCallback):
    """Collect average DQN loss per episode."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_losses = []
        self.current_losses = []

    def _on_step(self) -> bool:
        loss = self.model.logger.name_to_value.get("train/loss")
        if loss is not None:
            self.current_losses.append(loss)

        # Detect new episode (Monitor wrapper logs episode end in info)
        info = self.locals.get("infos", [{}])[0]
        if "episode" in info:  # episode ended
            if self.current_losses:
                self.episode_losses.append(np.mean(self.current_losses))
            self.current_losses = []  # reset for next episode

        return True

    def save(self):
        if not self.episode_losses:
            print("⚠️ No episode loss values tracked.")
            return

        pd.DataFrame({"episode_loss": self.episode_losses}).to_csv(LOSS_CSV, index=False)

        plt.figure(figsize=(8, 4))
        plt.plot(self.episode_losses, marker="o", alpha=0.8, color="red")
        plt.title("📉 Average DQN Loss Per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Avg Loss")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "loss_plot.png"))
        print("💾 Saved episode loss logs & plot!")



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
    env = Monitor(env)
    env.reset(seed=seed)
    return env

best_reward_global = -9999 # global best episode recorder
# ---------- TRAINING ----------
def train_dqn(total_steps, stage_size, lr):
    env, eval_env = make_env(0), make_env(100)

    # Load or create model
    if os.path.exists(MODEL_PATH):
        print(f"📦 Loading model from {MODEL_PATH}")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        model = DQN(
            "MlpPolicy", env,
            learning_rate=lr, buffer_size=500_000, batch_size=128,
            tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1,
            target_update_interval=500, exploration_fraction=0.4,
            exploration_final_eps=0.05, verbose=1, seed=0,
            tensorboard_log="./tb_dqn_lunarlander_v3",
            device=DEVICE, policy_kwargs=dict(net_arch=[256, 256]),
        )

    loss_logger = LossLogger()
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=MODEL_DIR,
        log_path=MODEL_DIR, eval_freq=20_000,
        deterministic=True, verbose=0
    )

    reward_progress = []
    stages = total_steps // stage_size

    global best_reward_global

    for s in range(stages):
        print(f"\n=== 🧠 Stage {s+1}/{stages} → {stage_size:,} steps ===")
        model.learn(stage_size, reset_num_timesteps=False,
                    callback=[eval_callback, loss_logger])
        model.save(MODEL_PATH)

        mean_r, std_r = evaluate_policy(model, eval_env, 10)
        reward_progress.append(mean_r)
        print(f"📈 Eval: mean={mean_r:.2f} ± {std_r:.2f}")

        if mean_r > best_reward_global:
            print(f"🎥 NEW BEST AVERAGE REWARD ({mean_r}) → Recording episode")
            best_reward_global = mean_r
            record_best_video(model)

    # Save loss plots
    loss_logger.save()

    # Save Reward CSV + Plot
    pd.DataFrame({"reward": reward_progress}).to_csv(REWARD_CSV, index=False)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, stages + 1), reward_progress, marker="o")
    plt.title("📈 Mean Reward During Training")
    plt.xlabel("Training Stage")
    plt.ylabel("Mean Reward")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_reward_plot.png"))
    print("💾 Saved reward logs & plot!")

    env.close(); eval_env.close()
    return model

# ---------- RECORD BEST VIDEO ----------
def record_best_video(model):
    env = make_env(1234, record=True, tag="best")
    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    env.close()


# ---------- EVALUATION (PRINT + PLOT) ----------
def evaluate_and_report(model, n_eps=20):
    env = make_env(999)
    outcomes = {"LANDED_OK": 0, "CRASH": 0,
                "OUT_OF_BOUNDS": 0, "ASLEEP": 0,
                "UNKNOWN": 0, "DONE": 0}
    rewards = []

    for ep in range(n_eps):
        obs, _ = env.reset()
        ep_reward = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if term or trunc:
                reason = termination_reason(env, obs)
                outcomes[reason] += 1
                rewards.append(ep_reward)
                break

    env.close()

    # 📌 DO NOT REMOVE: PRINT SUMMARY
    print("\n=== Evaluation Summary ===")
    for k, v in outcomes.items():
        print(f"{k:>15}: {v}")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    # Bar Plot
    plt.figure(figsize=(6, 4))
    plt.bar(list(outcomes.keys()), list(outcomes.values()), color="lightgreen")
    plt.title("🚀 LunarLander Episode Outcomes")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "evaluation_summary.png"))
    print("💾 Saved evaluation outcome bar graph!")


# ---------- MAIN ----------
if __name__ == "__main__":
    print(f"🚀 Training on {ENV_NAME} | Steps={args.total_steps:,} | LR={args.lr}")
    model = train_dqn(args.total_steps, args.stage_size, args.lr)
    evaluate_and_report(model, 20)
    print("🎉 Done!")
    print("🎉 Done! Video saved in:", VIDEO_DIR)
