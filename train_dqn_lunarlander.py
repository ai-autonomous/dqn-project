"""
DQN training script for LunarLander-v3 using Stable-Baselines3 (SB3)
Now supports CLI inputs for total_steps, stage_size, and learning_rate.
Generates reward progression and evaluation summary plots.
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
parser = argparse.ArgumentParser(description="Train DQN on LunarLander-v3 with configurable params.")
parser.add_argument("--total_steps", type=int, default=2_000_000, help="Total training timesteps")
parser.add_argument("--stage_size", type=int, default=200_000, help="Steps per training stage")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for DQN")
args = parser.parse_args()

# ---------- Environment ----------
ENV_NAME = "LunarLander-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Paths ----------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_lunarlander_v3.zip")
TB_LOG = "./tb_dqn_lunarlander_v3"
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"🚀 Starting DQN training on {ENV_NAME}")
print(f"🧠 Device: {DEVICE}")
print(f"💡 Total steps: {args.total_steps}, Stage size: {args.stage_size}, LR: {args.lr}")

# ---------- Helper: detect termination reason ----------
def termination_reason(env, obs):
    """Detect why the episode ended in LunarLander-v3."""
    um = env.unwrapped
    try:
        x_pos = float(obs[0])
        game_over = getattr(um, "game_over", False)
        awake = bool(um.lander.awake)
        if game_over:
            return "CRASH"
        elif abs(x_pos) >= 2.5:
            return "OUT_OF_BOUNDS"
        elif not awake:
            left, right = int(obs[6]), int(obs[7])
            return "LANDED_OK" if left == 1 and right == 1 else "ASLEEP"
        else:
            return "UNKNOWN"
    except Exception:
        return "DONE"


# ---------- Environment factory ----------
def make_env(seed=0):
    env = gym.make(ENV_NAME)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ---------- Training ----------
def train_dqn(total_steps, stage_size, lr):
    env = make_env(0)
    eval_env = make_env(100)
    policy_kwargs = dict(net_arch=[256, 256])

    # Load or initialize model
    if os.path.exists(MODEL_PATH):
        print(f"📦 Loading existing model from {MODEL_PATH}")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        print(f"🚀 Creating new DQN model for {ENV_NAME}")
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=lr,
            buffer_size=500_000,
            batch_size=128,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=500,
            exploration_fraction=0.4,
            exploration_final_eps=0.05,
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
        eval_freq=20_000,
        deterministic=True,
        render=False,
        verbose=0,
    )

    stages = total_steps // stage_size
    reward_progress = []

    for s in range(stages):
        print(f"\n=== 🧠 Stage {s+1}/{stages} → training {stage_size:,} steps ===")
        model.learn(total_timesteps=stage_size, reset_num_timesteps=False, callback=eval_callback)
        model.save(MODEL_PATH)
        print(f"💾 Saved checkpoint after {stage_size*(s+1):,} total steps")

        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10)
        reward_progress.append(mean_r)
        print(f"📈 Evaluation after {stage_size*(s+1):,} steps: mean={mean_r:.2f} ± {std_r:.2f}")

    # Plot training reward trend
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, stages + 1), reward_progress, marker="o")
    plt.title("📈 DQN Training Progress on LunarLander-v3")
    plt.xlabel("Training Stage")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_reward_plot.png"))
    print(f"📊 Saved training progress plot to {MODEL_DIR}/training_reward_plot.png")

    print("✅ Training complete!")
    env.close()
    eval_env.close()
    return model


# ---------- Evaluation ----------
def evaluate_and_report(model, n_eval_episodes=20, render=False):
    env = make_env(999)
    results = {"LANDED_OK": 0, "CRASH": 0, "OUT_OF_BOUNDS": 0, "ASLEEP": 0, "UNKNOWN": 0, "DONE": 0}
    rewards = []

    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        ep_reward, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if render:
                env.render()
            if terminated or truncated:
                reason = termination_reason(env, obs)
                results[reason] = results.get(reason, 0) + 1
                rewards.append(ep_reward)
                print(f"Ep {ep+1:02d}/{n_eval_episodes} → Reward={ep_reward:7.2f}, Steps={steps:3d}, End={reason}")
                break

    env.close()
    print("\n=== Evaluation Summary ===")
    for k, v in results.items():
        print(f"{k:>15}: {v}")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    # Plot evaluation summary
    labels = list(results.keys())
    counts = list(results.values())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts, color="skyblue")
    plt.title("🚀 Evaluation Results (Episode Outcomes)")
    plt.ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "evaluation_summary.png"))
    print(f"📊 Saved evaluation summary plot to {MODEL_DIR}/evaluation_summary.png")


# ---------- Main ----------
if __name__ == "__main__":
    model = train_dqn(args.total_steps, args.stage_size, args.lr)
    evaluate_and_report(model, n_eval_episodes=20, render=False)
