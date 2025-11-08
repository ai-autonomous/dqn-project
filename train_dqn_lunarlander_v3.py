"""
DQN training for LunarLander-v3 using Stable-Baselines3 (SB3)
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch
import matplotlib.pyplot as plt


# ---------- Termination reason helper ----------
def termination_reason(env, obs):
    um = env.unwrapped
    x_pos = float(obs[0])
    game_over = getattr(um, "game_over", False)
    try:
        awake = bool(um.lander.awake)
    except Exception:
        awake = True

    # v3 has slightly larger viewport (~2.5 instead of 1.0)
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


# ---------- Hyperparameters ----------
TOTAL_TIMESTEPS = 500_000
MODEL_DIR = "models_v3"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_lunarlander_v3")


# ---------- Training ----------
def train():
    env = make_env(seed=0)
    policy_kwargs = dict(net_arch=[256, 256])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        verbose=1,
        seed=0,
        tensorboard_log="./tb_dqn_lunar_v3",
        policy_kwargs=policy_kwargs,
        device="auto" if torch.cuda.is_available() else "cpu",
    )

    eval_env = make_env(seed=100)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=MODEL_DIR,
        log_path=MODEL_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}.zip")

    env.close()
    eval_env.close()
    return model


# ---------- Evaluation ----------
def evaluate_and_report(model, n_eval_episodes=20, render=False):
    env = make_env(seed=999)
    counts = {"LANDED_OK": 0, "CRASH": 0, "OUT_OF_BOUNDS": 0, "ASLEEP": 0, "UNKNOWN": 0}
    rewards = []

    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        done, ep_reward, steps = False, 0.0, 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if render:
                env.render()
            if terminated or truncated:
                reason = termination_reason(env, obs)
                counts[reason] = counts.get(reason, 0) + 1
                rewards.append(ep_reward)
                print(f"Episode {ep+1}/{n_eval_episodes} → Reward={ep_reward:.1f}, Steps={steps}, End={reason}")
                break

    env.close()

    print("\n=== Evaluation Summary ===")
    for k, v in counts.items():
        print(f"{k:>15}: {v}")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    plt.hist(rewards, bins=20)
    plt.title("Evaluation Reward Distribution (LunarLander-v3)")
    plt.xlabel("Episode Reward")
    plt.ylabel("Count")
    plt.show()


# ---------- Main ----------
if __name__ == "__main__":
    model = train()
    evaluate_and_report(model, n_eval_episodes=20, render=False)
