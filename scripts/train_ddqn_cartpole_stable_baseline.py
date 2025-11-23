"""
Improved Double DQN training script for CartPole-v1 using Stable-Baselines3 (SB3).
Now includes advanced termination reasoning similar to LunarLander:
  - SOLVED: Pole balances full max steps
  - GOOD_RUN: Balance exceeds strong threshold (≥195 steps)
  - FAIL: Pole fell or went out of bounds before threshold
  - TIME_LIMIT: Episode ended due to time truncation (not failure)

Generates reward progression & evaluation summary plots.
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ---------- Parse CLI Arguments ----------
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
MODEL_PATH = os.path.join(MODEL_DIR, "double_dqn_cartpole_v1.zip")
TB_LOG = "./tb_double_dqn_cartpole_v1"
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Starting Double DQN training on {ENV_NAME}")
print(f"Device: {DEVICE}")
print(f"Total steps: {args.total_steps}, Stage size: {args.stage_size}, LR: {args.lr}")

# ===================== TERMINATION REASON (ADVANCED) =====================
def termination_reason(terminated, truncated, steps, max_steps):
    if truncated and steps == max_steps:
        return "SOLVED"           # Balanced entire max steps
    elif truncated:
        return "TIME_LIMIT"       # Episode timeout, but not solved fully
    elif terminated and steps >= 195:
        return "GOOD_RUN"         # Survived long enough (classic threshold)
    elif terminated:
        return "FAIL"             # Fell early
    else:
        return "UNKNOWN"


# ---------- Environment factory ----------
def make_env(seed=0):
    env = gym.make(ENV_NAME)
    env = Monitor(env)
    env.reset(seed=seed)
    return env

# ---------- Training ----------
def train_double_dqn(total_steps, stage_size, lr):
    env = make_env(0)
    eval_env = make_env(100)
    policy_kwargs = dict(net_arch=[128, 128])

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        print(f"Creating new Double DQN model for {ENV_NAME}")
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

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=MODEL_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=0,
    )

    stages = max(1, total_steps // stage_size)
    reward_progress = []

    for s in range(stages):
        print(f"\n=== Stage {s+1}/{stages} → training {stage_size:,} steps ===")
        model.learn(total_timesteps=stage_size, reset_num_timesteps=False, callback=eval_callback)
        model.save(MODEL_PATH)
        print(f"Saved checkpoint after {stage_size*(s+1):,} total steps")

        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10)
        reward_progress.append(mean_r)
        print(f"Evaluation after {stage_size*(s+1):,} steps: mean={mean_r:.2f} ± {std_r:.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, stages + 1), reward_progress, marker="o")
    plt.title("Double DQN Training Progress on CartPole-v1")
    plt.xlabel("Training Stage")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_reward_plot_double_dqn_cartpole.png"))
    print(f"Saved training progress plot to {MODEL_DIR}/training_reward_plot_double_dqn_cartpole.png")

    env.close(); eval_env.close()
    return model

# ---------- Evaluation ----------
def evaluate_and_report(model, n_eval_episodes=20, render=False):
    env = make_env(999)
    max_steps = env.env.spec.max_episode_steps  # dynamic limit

    results = {"SOLVED": 0, "GOOD_RUN": 0, "FAIL": 0, "TIME_LIMIT": 0, "UNKNOWN": 0}
    rewards = []

    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        ep_reward, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if render: env.render()
            if terminated or truncated:
                reason = termination_reason(terminated, truncated, steps, max_steps)
                results[reason] += 1
                rewards.append(ep_reward)
                print(f"Ep {ep+1:02d}/{n_eval_episodes} → Reward={ep_reward:7.2f}, Steps={steps:3d}, End={reason}")
                break

    env.close()

    print("\n=== Evaluation Summary ===")
    for k, v in results.items():
        print(f"{k:>12}: {v}")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    # Plot summary
    labels = list(results.keys())
    counts = list(results.values())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color="skyblue")
    plt.title("Evaluation Results — Double DQN CartPole-v1")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "evaluation_summary_double_dqn_cartpole.png"))
    print(f"Saved evaluation summary plot to {MODEL_DIR}/evaluation_summary_double_dqn_cartpole.png")

# ---------- Main ----------
if __name__ == "__main__":
    model = train_double_dqn(args.total_steps, args.stage_size, args.lr)
    evaluate_and_report(model, n_eval_episodes=20, render=False)
