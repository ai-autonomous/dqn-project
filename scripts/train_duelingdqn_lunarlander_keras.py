# DDQN LunarLander Implementation - Keras
import os
import argparse
import random
import time
from pathlib import Path
from typing import Tuple
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import matplotlib.pyplot as plt

from gymnasium.wrappers import RecordVideo

# ========= CLI Inputs (OPTIMIZED) =========
parser = argparse.ArgumentParser(description="Train Double DQN on LunarLander-v3")
parser.add_argument("--env", type=str, default="LunarLander-v3")
# Increased total steps for more robust training within the 30 min wall-time limit
parser.add_argument("--total_steps", type=int, default=500_000)
parser.add_argument("--stage_size", type=int, default=50_000)
# Learning rate is kept low (1e-4) for stability
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--save_path", type=str, default="ddqn_LunarLander_keras")
# Increased buffer size for better experience decorrelation
parser.add_argument("--buffer_size", type=int, default=200_000)
parser.add_argument("--eps_start", type=float, default=1.0)
parser.add_argument("--eps_end", type=float, default=0.01)
# Slower decay to allow crucial early exploration
parser.add_argument("--eps_decay_frames", type=int, default=120_000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
# Target network sync frequency is kept reasonable
parser.add_argument("--sync_steps", type=int, default=1500)
parser.add_argument("--eval_every", type=int, default=50)
parser.add_argument("--min_buffer", type=int, default=20000)
parser.add_argument("--train_every", type=int, default=1)
parser.add_argument("--record_best_video", action="store_true", help="Record video when avg100 >= 200")
args = parser.parse_args()

# ========= Reproducibility =========
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

MODEL_DIR = Path("models")
VIDEO_DIR = os.path.join(MODEL_DIR, "best_video")
#PLOTS_DIR = Path("plots") # Added PLOTS_DIR for cleaner saving
MODEL_DIR.mkdir(parents=True, exist_ok=True)
#PLOTS_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
MODEL_FILE = MODEL_DIR / "ddqn_lunarlander_keras.pth"
LOG_FILE = Path("ddqn_lunarlander_keras_results.txt")
MODEL_PATH = os.path.join(MODEL_DIR, "dqn_lunarlander_v3.keras")
LOSS_CSV = os.path.join(MODEL_DIR, "loss_log.csv")
REWARD_CSV = os.path.join(MODEL_DIR, "reward_log.csv")

# ----------------- Utilities -----------------
def make_env(seed=0, record=False, tag="best"):
    env = gym.make(args.env, render_mode="rgb_array" if record else None)
    if record:
        env = RecordVideo(env, str(VIDEO_DIR), name_prefix=f"best_{tag}", episode_trigger=lambda x: x == 1) # Only record one episode
    #env = Monitor(env)
    env.reset(seed=seed)
    return env


def termination_reason(env, obs):
    # best effort, robust to gymnasium changes
    um = getattr(env, "unwrapped", None)
    try:
        x_pos = float(obs[0])
        if hasattr(um, "game_over") and bool(getattr(um, "game_over", False)):
            return "CRASH"
        elif abs(x_pos) >= 2.5:
            return "OUT_OF_BOUNDS"
        # legs contacts: obs[6], obs[7] if present
        left = int(obs[6]) if len(obs) > 6 else 0
        right = int(obs[7]) if len(obs) > 7 else 0
        # sleeping/landed heuristic via velocity/angle
        if left == 1 and right == 1:
            return "LANDED_OK"
        return "UNKNOWN"
    except Exception:
        return "DONE"

# ========= Replay Buffer =========
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return (s.astype(np.float32),
                a.astype(np.int32),
                r.astype(np.float32),
                ns.astype(np.float32),
                d.astype(np.float32))

    def __len__(self):
        return len(self.buffer)

# ========= Q Network (Pure DDQN: non-dueling) - INCREASED CAPACITY =========
def build_q_network(state_dim, action_dim):
    inputs = layers.Input(shape=(state_dim,), dtype=tf.float32)
    # Increased the second layer to 256 neurons for better function approximation
    x = layers.Dense(256, activation="relu", kernel_initializer='he_uniform')(inputs)
    x = layers.Dense(256, activation="relu", kernel_initializer='he_uniform')(x)
    outputs = layers.Dense(action_dim, activation=None, kernel_initializer='he_uniform')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# ----------------- Agent -----------------
class DDQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, batch_size):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.online = build_q_network(state_dim, action_dim)
        self.target = build_q_network(state_dim, action_dim)
        self.target.set_weights(self.online.get_weights())

        self.optimizer = optimizers.Adam(learning_rate=lr)
        # Huber Loss is already correctly used for stability!
        self.loss_fn = losses.Huber()

    def act(self, obs, eps):
        if random.random() < eps:
            return random.randrange(self.action_dim)
        q = self.online(np.array([obs], dtype=np.float32))
        return int(tf.argmax(q[0]).numpy())

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        # Double DQN targets calculation logic is correct

        # Find the best action a' from the ONLINE network
        next_q_online = self.online(next_states)
        next_actions = tf.argmax(next_q_online, axis=1)

        # Use the TARGET network to get the Q-value for that best action a'
        next_q_target = self.target(next_states)
        
        # Gather the Q-value for the action selected by the online network
        next_q_for_best_action = tf.gather_nd(next_q_target, 
                                              tf.stack([tf.range(self.batch_size), 
                                                        tf.cast(next_actions, tf.int32)], axis=1))
        
        target_q = rewards + self.gamma * next_q_for_best_action * (1.0 - dones)

        # Train online network
        with tf.GradientTape() as tape:
            q_values = self.online(states)                               # (B, A)
            action_mask = tf.one_hot(actions, self.action_dim, dtype=tf.float32)
            chosen_q = tf.reduce_sum(q_values * action_mask, axis=1)     # (B,)
            loss = self.loss_fn(target_q, chosen_q)

        grads = tape.gradient(loss, self.online.trainable_variables)
        # Gradient clipping for stability
        grads = [tf.clip_by_norm(g, 10.0) if g is not None else None for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.online.trainable_variables))
        return loss

    def call_train_step(self, replay: ReplayBuffer):
        if len(replay) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = replay.sample(self.batch_size)
        
        # Convert samples to TF tensors before passing to tf.function
        states_t = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_t = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_t = tf.convert_to_tensor(dones, dtype=tf.float32)

        loss = self.train_step(states_t, actions_t, rewards_t, next_states_t, dones_t)
        return float(loss.numpy())


    def update_target(self):
        self.target.set_weights(self.online.get_weights())

    def save(self, path: Path):
        self.online.save(str(path))

    def load(self, path: Path):
        self.online = tf.keras.models.load_model(str(path))
        self.target.set_weights(self.online.get_weights())


# ----------------- Main Training -----------------
def train(args):
    # Env and dims
    env = gym.make(args.env)
    PLOTS_DIR = Path("plots") # Added PLOTS_DIR for cleaner saving
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        reset_ret = env.reset(seed=args.seed)
        obs0 = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
    except TypeError:
        reset_ret = env.reset()
        obs0 = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DDQNAgent(state_dim, action_dim, lr=args.lr, gamma=args.gamma, batch_size=args.batch_size)
    replay = ReplayBuffer(capacity=args.buffer_size)

    # Warmup
    obs = obs0
    for _ in range(args.min_buffer):
        a = env.action_space.sample()
        step_ret = env.step(a)
        if len(step_ret) == 5:
            next_obs, r, term, trunc, info = step_ret
            done = bool(term or trunc)
        else:
            next_obs, r, done, info = step_ret
        replay.push(obs, a, r, next_obs, float(done))
        obs = next_obs if not done else env.reset(seed=args.seed)[0]

    print(f"Warmup complete: replay size = {len(replay)}")

    # Metrics
    ALL_REWARDS = []
    ALL_LOSSES = []
    best_avg100 = -np.inf
    frames = 0
    episodes = 0
    start = time.time()

    # Training
    while episodes < args.episodes and frames < args.total_steps:
        reset_ret = env.reset(seed=args.seed + episodes)
        state = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        ep_reward = 0.0

        while True and frames < args.total_steps:
            frames += 1
            # epsilon schedule (exponential w/ floor)
            # Slower decay schedule allows for better exploration
            frac = min(1.0, frames / args.eps_decay_frames)
            eps = args.eps_end + (args.eps_start - args.eps_end) * np.exp(-3.0 * frac)

            action = agent.act(state, eps)
            step_ret = env.step(action)
            if len(step_ret) == 5:
                next_state, reward, term, trunc, info = step_ret
                done = bool(term or trunc)
            else:
                next_state, reward, done, info = step_ret

            replay.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            if frames % args.train_every == 0:
                # Use call_train_step to wrap numpy conversion for the tf.function
                loss = agent.call_train_step(replay)
                if loss is not None:
                    ALL_LOSSES.append(loss)

            if frames % args.sync_steps == 0:
                agent.update_target()

            if done:
                break

        ALL_REWARDS.append(ep_reward)
        episodes += 1

        avg100 = float(np.mean(ALL_REWARDS[-100:]))
        
        # Calculate elapsed time and project total time
        elapsed = (time.time() - start) / 60.0
        

        print(f"Ep {episodes:4d} | Frame {frames:7d} | EpR {ep_reward:8.2f} | Avg100 {avg100:8.2f} | eps~{eps:.3f} | Time {elapsed:.2f} min")

        # Save best & optionally record
        if avg100 > best_avg100:
            best_avg100 = avg100
            agent.save(MODEL_PATH)
            print(f"New best avg100 {best_avg100:.2f} — model saved at {MODEL_PATH}")
            if args.record_best_video and avg100 >= 200.0:
                print("Recording best video...")
                record_best_video(agent, tag="best")

        # Append log (FIXED: using LOG_FILE instead of MODEL_PATH)
        with open(LOG_FILE, "a") as f:
            f.write(f"Ep {episodes} Frame {frames} Reward {ep_reward:.2f} Avg100 {avg100:.2f}\n")

    env.close()
    elapsed = (time.time() - start) / 60.0
    print(f"Training finished. Frames={frames}, Episodes={episodes}, Time={elapsed:.2f} min")

    # Final save
    agent.save(MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}")

    # Plots
    plt.figure(figsize=(8, 4))
    plt.plot(ALL_REWARDS, alpha=0.8, color="blue")
    # Plot moving average
    moving_avg = np.convolve(ALL_REWARDS, np.ones(100)/100, mode='valid')
    plt.plot(np.arange(len(moving_avg)) + 100, moving_avg, color='red', label='Avg 100 Episodes')
    plt.title("Reward Per Episode - DDQN LunarLander (Keras)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reward_plot.png")

    if ALL_LOSSES:
        avg_loss = np.mean(ALL_LOSSES)
        print(f"Average Training Loss: {avg_loss:.5f}")
        plt.figure(figsize=(8, 4))
        # Plot smoothed loss
        loss_window = max(1, len(ALL_LOSSES) // 1000)
        smoothed_losses = np.convolve(ALL_LOSSES, np.ones(loss_window)/loss_window, mode='valid')
        plt.plot(smoothed_losses, alpha=0.8, color="red")
        plt.title("Smoothed Loss Curve - DDQN LunarLander (Keras)")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "loss_plot.png")
    else:
        print("No training steps were executed, so average loss is N/A.")
    print(f"Saved plots to {PLOTS_DIR}")
    return agent, ALL_REWARDS, ALL_LOSSES

# ---------- RECORD BEST VIDEO ----------
def record_best_video(agent: DDQNAgent, tag="best"):
    env = make_env(seed=args.seed, record=True, tag=tag)
    obs, _ = env.reset()
    while True:
        q_vals = agent.online(np.array([obs], dtype=np.float32))
        action = int(tf.argmax(q_vals[0]).numpy())
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    env.close()

# ----------------- EVALUATION (PRINT SUMMARY) -----------------
def evaluate_and_report(agent: DDQNAgent, n_eps: int = 20):
    env = gym.make(args.env)
    outcomes = {"LANDED_OK": 0, "CRASH": 0, "OUT_OF_BOUNDS": 0, "UNKNOWN": 0, "DONE": 0}
    rewards = []

    for _ in range(n_eps):
        reset_ret = env.reset(seed=args.seed)
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        ep_r = 0.0
        while True:
            q_vals = agent.online(np.array([obs], dtype=np.float32))
            action = int(tf.argmax(q_vals[0]).numpy())
            step_ret = env.step(action)
            if len(step_ret) == 5:
                obs, reward, term, trunc, info = step_ret
                done = bool(term or trunc)
            else:
                obs, reward, done, info = step_ret
            ep_r += reward
            if done:
                reason = termination_reason(env, obs)
                outcomes[reason] = outcomes.get(reason, 0) + 1
                rewards.append(ep_r)
                break
    env.close()

    print("=== Evaluation Summary ===")
    for k, v in outcomes.items():
        print(f"{k:>15}: {v}")
    print(f"Mean Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

    # Plot outcomes
    labels, counts = list(outcomes.keys()), list(outcomes.values())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts)
    plt.title("LunarLander Episode Outcomes")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "evaluation_summary.png")
    print(f"Saved evaluation plot to {MODEL_DIR / 'evaluation_summary.png'}")
	
# ----------------- Run -----------------
if __name__ == "__main__":
    print(f"🚀 Training DDQN (Keras) Episodes={args.episodes} | Frames={args.total_steps:,} | LR={args.lr}")
    agent, rewards, losses = train(args)
    # Load best for evaluation, if saved
    if os.path.exists(MODEL_PATH):
        try:
            agent.load(MODEL_PATH)
        except Exception as e:
            print(f"Could not load best model for final evaluation: {e}")
    evaluate_and_report(agent, n_eps=20)
    print("🎉 Done!")