"""
Improved Double DQN training script for CartPole-v1 using Stable-Baselines3 (SB3).
Includes:
  • Advanced termination reasoning (SOLVED, GOOD_RUN, FAIL, etc.)
  • Best-episode video recording only once (saved in models/video/)
  • Reward progression & evaluation plots
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ---------- CLI Args ----------
parser = argparse.ArgumentParser(description="Train Double DQN on CartPole-v1 with configurable params.")
parser.add_argument("--total_steps", type=int, default=1_000_000, help="Total training timesteps")
parser.add_argument("--stage_size", type=int, default=100_000, help="Steps per training stage")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for Double DQN")
args = parser.parse_args()

# ---------- Environment ----------
ENV_NAME = "CartPole-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Paths ----------
MODEL_DIR = "models"
VIDEO_DIR = os.path.join(MODEL_DIR, "video")
MODEL_PATH = os.path.join(MODEL_DIR, "double_dqn_cartpole_v1.zip")
TB_LOG = "./tb_double_dqn_cartpole_v1"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

print(f"Starting Double DQN training on {ENV_NAME}")
print(f"Device: {DEVICE}")
print(f"Total steps: {args.total_steps}, Stage size: {args.stage_size}, LR: {args.lr}")

# ===================== TERMINATION REASON =====================
def termination_reason(terminated, truncated, steps, max_steps):
    if truncated and steps == max_steps:
        return "SOLVED"
    elif truncated:
        return "TIME_LIMIT"
    elif terminated and steps >= 195:
        return "GOOD_RUN"
    elif terminated:
        return "FAIL"
    else:
        return "UNKNOWN"

# ---------- Environment Factory ----------
def make_env(seed=0, record=False, video_name="best_cartpole"):
    env = gym.make(ENV_NAME, render_mode="rgb_array" if record else None)
    if record:
        env = RecordVideo(env, VIDEO_DIR, name_prefix=video_name)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

# ---------- BEST VIDEO RECORDER ----------
best_reward = -999999  # global tracker

def record_best_video(model):
    global best_reward
    env = make_env(seed=777, record=True, video_name="best_cartpole_episode")
    obs, _ = env.reset()
    total_r, steps = 0, 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += reward; steps += 1
        if terminated or truncated:
            break

    env.close()

    # Only save video if performance improved
    if total_r > best_reward:
        best_reward = total_r
        print(f"🎥 New best score! Saved video: reward={total_r:.2f}, steps={steps}")
    else:
        # delete incorrect/latest video
        for f in os.listdir(VIDEO_DIR):
            if "best_cartpole_episode" in f:
                os.remove(os.path.join(VIDEO_DIR, f))

# ---------- Training ----------
def train_double_dqn(total_steps, stage_size, lr):
    env, eval_env = make_env(0), make_env(100)
    policy_kwargs = dict(net_arch=[128, 128])

    if os.path.exists(MODEL_PATH):
        print(f"📦 Loading model from {MODEL_PATH}")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        print(f"🆕 Creating new Double DQN model for {ENV_NAME}")
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=lr,
            buffer_size=50_000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=250,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            verbose=1,
            seed=0,
            tensorboard_log=TB_LOG,
            policy_kwargs=policy_kwargs,
            device=DEVICE,
        )

    stages = max(1, total_steps // stage_size)
    reward_progress = []

    for s in range(stages):
        print(f"\n=== 🚀 Stage {s+1}/{stages} → training {stage_size:,} steps ===")
        model.learn(stage_size, reset_num_timesteps=False)
        model.save(MODEL_PATH)

        mean_r, std_r = evaluate_policy(model, eval_env, 10)
        reward_progress.append(mean_r)
        print(f"📈 Eval: mean={mean_r:.2f} ± {std_r:.2f}")

        # 🎥 Try recording best video after each stage
        record_best_video(model)

    # Plot training rewards
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, stages + 1), reward_progress, marker="o")
    plt.title("Double DQN Training Progress on CartPole-v1")
    plt.xlabel("Training Stage"); plt.ylabel("Mean Reward")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_reward_plot_double_dqn_cartpole.png"))

    env.close(); eval_env.close()
    return model

# ---------- Evaluation ----------
def evaluate_and_report(model, n_eval_episodes=20, render=False):
    env = make_env(999)
    max_steps = env.env.spec.max_episode_steps

    results = {"SOLVED": 0, "GOOD_RUN": 0, "FAIL": 0, "TIME_LIMIT": 0, "UNKNOWN": 0}
    rewards = []

    for ep in range(n_eval_episodes):
        obs, _ = env.reset()
        ep_reward, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward; steps += 1
            if render: env.render()
            if terminated or truncated:
                outcome = termination_reason(terminated, truncated, steps, max_steps)
                results[outcome] += 1; rewards.append(ep_reward)
                break
    env.close()

    print("\n=== 🧪 Evaluation Summary ===")
    for k, v in results.items(): print(f"{k:>12}: {v}")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

# ---------- MAIN ----------
if __name__ == "__main__":
    model = train_double_dqn(args.total_steps, args.stage_size, args.lr)
    evaluate_and_report(model, 20)
    print("🎉 Finished!")
