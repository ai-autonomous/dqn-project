"""
Improved DQN training for LunarLander-v3 using Stable-Baselines3 (SB3)
Supports staged training + evaluation summaries.
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ---------- Helper: detect termination reason ----------
def termination_reason(env, obs):
    um = env.unwrapped
    x_pos = float(obs[0])
    game_over = getattr(um, "game_over", False)
    try:
        awake = bool(um.lander.awake)
    except Exception:
        awake = True

    if game_over:
        return "CRASH"
    elif abs(x_pos) >= 2.5:
        return "OUT_OF_BOUNDS"
    elif not awake:
        left, right = int(obs[6]), int(obs[7])
        return "LANDED_OK" if left == 1 and right == 1 else "ASLEEP"
    else:
        return "UNKNOWN"

# ---------- Environment factory ----------
def make_env(seed=0):
    env = gym.make("LunarLander-v3")
    env = Monitor(env)
    env.reset(seed=seed)
    return env

# ---------- Paths ----------
MODEL_DIR = "models_v3"
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_lunarlander_v3.zip")
TB_LOG = "./tb_dqn_lunar_v3"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Training function ----------
def train_dqn(total_steps=2_000_000, stage_size=200_000):
    env = make_env(0)
    eval_env = make_env(100)
    policy_kwargs = dict(net_arch=[256, 256])

    # either load existing model or create a new one
    if os.path.exists(MODEL_PATH):
        print(f"📦 Loading existing model from {MODEL_PATH}")
        model = DQN.load(MODEL_PATH, env=env)
    else:
        print("🚀 Starting new DQN training run")
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=5e-4,
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
            device="cuda" if torch.cuda.is_available() else "cpu",
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
def evaluate_and_report(model, n_eval_episodes=30, render=False):
    env = make_env(999)
    results = {"LANDED_OK": 0, "CRASH": 0, "OUT_OF_BOUNDS": 0, "ASLEEP": 0, "UNKNOWN": 0}
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
                results[reason] += 1
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
    model = train_dqn(total_steps=2_000_000, stage_size=200_000)
    evaluate_and_report(model, n_eval_episodes=30, render=False)
