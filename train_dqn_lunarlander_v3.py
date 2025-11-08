"""
Configurable DQN training using Stable-Baselines3 (SB3)
Environment name is passed as input (e.g., LunarLander-v3 or CartPole-v1)
Supports staged training, checkpointing, and evaluation.
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
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


# ---------- CLI Arguments ----------
parser = argparse.ArgumentParser(description="Train DQN on a configurable Gymnasium environment.")
parser.add_argument("--env", type=str, default="LunarLander-v3", help="Gymnasium environment name")
parser.add_argument("--total_steps", type=int, default=2_000_000, help="Total training timesteps")
parser.add_argument("--stage_size", type=int, default=200_000, help="Steps per training stage")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for DQN")
args = parser.parse_args()


# ---------- Global Config ----------
ENV_NAME = args.env
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models_v3"
MODEL_BASENAME = f"dqn_{ENV_NAME.replace('-', '_').lower()}"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_BASENAME}.zip")
TB_LOG = f"./tb_{MODEL_BASENAME}"
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"🚀 Starting DQN training for environment: {ENV_NAME}")
print(f"🧠 Device: {DEVICE}")
print(f"💾 Model path: {MODEL_PATH}")
print(f"📊 TensorBoard log dir: {TB_LOG}")


# ---------- Termination Reason Helper ----------
def termination_reason(env, obs):
    """Detect termination reason for environments like LunarLander."""
    um = env.unwrapped
    try:
        x_pos = float(obs[0])
        game_over = getattr(um, "game_over", False)
        awake = bool(getattr(um.lander, "awake", True))
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


# ---------- Environment Factory ----------
def make_env(seed=0):
    env = gym.make(ENV_NAME)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ---------- Training ----------
def train_dqn(total_steps=args.total_steps, stage_size=args.stage_size, lr=args.lr):
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
    for s in range(stages):
        print(f"\n=== 🧠 Stage {s+1}/{stages} → training {stage_size:,} steps ===")
        model.learn(total_timesteps=stage_size, reset_num_timesteps=False, callback=eval_callback)
        model.save(MODEL_PATH)
        print(f"💾 Saved checkpoint after {stage_size*(s+1):,} total steps")

        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"📈 Evaluation after {stage_size*(s+1):,} steps: mean={mean_r:.2f} ± {std_r:.2f}")

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


# ---------- Main ----------
if __name__ == "__main__":
    model = train_dqn()
    evaluate_and_report(model, n_eval_episodes=20, render=False)
