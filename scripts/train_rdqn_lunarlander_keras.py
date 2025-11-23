import random
from collections import deque
import os
import argparse

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import Box2D  # DO NOT REMOVE

# =========================
# Device Selection (CPU/GPU/Metal Auto)
# =========================
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    DEVICE = "/GPU:0"   # NVIDIA CUDA or Apple Metal (tensorflow-metal)
else:
    DEVICE = "/CPU:0"
print(f"🔧 Using TensorFlow device: {DEVICE}")

# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Hyperparameter Helpers
# =========================
def discount_rate():  # Gamma
    return 1

def batch_size():
    return 128

# =========================
# Environment
# =========================
envLunar = gym.make('LunarLander-v3')
envLunar.reset(seed=SEED)
envLunar.action_space.seed(SEED)
envLunar.observation_space.seed(SEED)

# =========================
# Prioritized Replay Buffer
# =========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights,
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# =========================
# Rainbow-Style DQN Agent
# =========================
class RainbowDQN:
    def __init__(self, states, actions, alpha, gamma,
                 epsilon, epsilon_min, epsilon_decay,
                 buffer_size=50000, n_step=3,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=100000):

        self.nS = states
        self.nA = actions
        self.gamma = gamma
        self.alpha = alpha

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = PrioritizedReplayBuffer(buffer_size, alpha=per_alpha)
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.frame_idx = 1

        self.n_step = n_step
        self.n_gamma = gamma ** n_step
        self.n_step_buffer = deque(maxlen=n_step)

        with tf.device(DEVICE):
            self.model = self.build_model()
            self.model_target = self.build_model()
            self.update_target_from_model()

        self.loss = []

    def build_model(self):
        inputs = keras.layers.Input(shape=(self.nS,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dense(128, activation='relu')(x)

        V = keras.layers.Dense(1)(x)
        A = keras.layers.Dense(self.nA)(x)

        A_mean = keras.layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(A)
        Q = keras.layers.Add()([V, keras.layers.Subtract()([A, A_mean])])

        model = keras.Model(inputs=inputs, outputs=Q)
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=self.alpha))
        return model

    def update_target_from_model(self):
        self.model_target.set_weights(self.model.get_weights())

    def _get_beta(self):
        return min(1.0, self.per_beta_start +
                   (1.0 - self.per_beta_start) * (self.frame_idx / float(self.per_beta_frames)))

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)
        with tf.device(DEVICE):
            q_vals = self.model.predict(state.reshape(1, -1), verbose=0)
        return int(np.argmax(q_vals[0]))

    def _store_n_step(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return None
        R = sum((self.gamma ** i) * t[2] for i, t in enumerate(self.n_step_buffer))
        s, a, _, _, _ = self.n_step_buffer[0]
        _, _, _, ns, d = self.n_step_buffer[-1]
        return (s, a, R, ns, d)

    def remember(self, state, action, reward, nstate, done):
        n = self._store_n_step((state, action, reward, nstate, done))
        if n:
            self.memory.push(*n)

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        beta = self._get_beta()
        self.frame_idx += 1

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, beta)

        with tf.device(DEVICE):
            q_st = self.model.predict(states, verbose=0)
            q_next_online = self.model.predict(next_states, verbose=0)
            q_next_target = self.model_target.predict(next_states, verbose=0)

        td_errors = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            a = actions[i]
            done = dones[i]
            r = rewards[i]
            target = q_st[i]
            if done:
                y = r
            else:
                best_a = np.argmax(q_next_online[i])
                y = r + (self.gamma ** self.n_step) * q_next_target[i][best_a]
            td_errors[i] = y - target[a]
            target[a] = y
            q_st[i] = target

        weights = weights.squeeze()

        with tf.device(DEVICE):
            hist = self.model.fit(states, q_st, epochs=1, verbose=0,
                                  batch_size=batch_size, sample_weight=weights)

        self.loss.append(hist.history['loss'][0])
        self.memory.update_priorities(indices, np.abs(td_errors) + 1e-6)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =========================
# Create and Train Rainbow Agent
# =========================

MODEL_DIR = "keras_rainbow_ddqn_models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "rainbow_lunarlander_v3.h5")

parser = argparse.ArgumentParser(description="Train Rainbow DQN on LunarLander-v3")
parser.add_argument("--total_steps", type=int, default=500)
parser.add_argument("--stage_size", type=int, default=500)
parser.add_argument("--lr", type=float, default=5e-4)
args = parser.parse_args()

EPISODES = args.total_steps
MAX_STEPS = args.stage_size
LR = args.lr
TARGET_SYNC_EVERY = 1
SOLVE_LINE = 200.0

nS = envLunar.observation_space.shape[0]
nA = envLunar.action_space.n

rainbow = RainbowDQN(nS, nA, LR, discount_rate(),
                     epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)

batch = batch_size()
rewards, epsilons, losses = [], [], []

for e in range(EPISODES):
    state, _ = envLunar.reset(seed=SEED + e)
    total_reward = 0.0

    for t in range(MAX_STEPS):
        action = rainbow.action(state)
        nstate, r, terminated, truncated, _ = envLunar.step(action)
        done = terminated or truncated

        rainbow.remember(state, action, r, nstate, done)
        rainbow.experience_replay(batch)

        state = nstate
        total_reward += r
        if done:
            break

    rewards.append(total_reward)
    epsilons.append(rainbow.epsilon)
    losses.append(rainbow.loss[-1] if rainbow.loss else np.nan)

    if (e + 1) % TARGET_SYNC_EVERY == 0:
        rainbow.update_target_from_model()

    print(f"[Episode {e+1}] reward={total_reward:.2f} epsilon={rainbow.epsilon:.3f} last_loss={losses[-1]:.6f}")

# Save Model
with tf.device(DEVICE):
    rainbow.model.save(MODEL_PATH)
print(f"\n💾 Saved Rainbow Model → {MODEL_PATH}")

# =========================
# TEST AND METRIC SUMMARY
# =========================
TEST_EPISODES = 20
test_rewards = []
metrics = {k: 0 for k in ["LANDED_OK", "CRASHED", "OUT_OF_BOUNDS", "ASLEEP", "UNKNOWN", "DONE"]}

for te in range(TEST_EPISODES):
    s, _ = envLunar.reset(seed=SEED + 1000 + te)
    ep_r, done = 0.0, False

    while not done:
        with tf.device(DEVICE):
            q_vals = rainbow.model.predict(s.reshape(1, -1), verbose=0)
        a = int(np.argmax(q_vals[0]))
        s, r, terminated, truncated, info = envLunar.step(a)
        done = terminated or truncated
        ep_r += r

    reason = info.get("termination_reason", "").upper()
    if "LAND" in reason: metrics["LANDED_OK"] += 1
    elif "CRASH" in reason: metrics["CRASHED"] += 1
    elif "OUT" in reason: metrics["OUT_OF_BOUNDS"] += 1
    elif "ASLEEP" in reason: metrics["ASLEEP"] += 1
    elif "DONE" in reason: metrics["DONE"] += 1
    else: metrics["UNKNOWN"] += 1

    test_rewards.append(ep_r)

print("\n==== 🧪 TEST SUMMARY ====")
for k, v in metrics.items(): print(f"{k}: {v}")
print(f"\nMean Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")

# =========================
# SAVE GRAPHS + CSV
# =========================
pd.DataFrame({"reward": rewards, "epsilon": epsilons, "loss": losses}).to_csv(
    os.path.join(MODEL_DIR, "training_data.csv"), index=False)

plt.figure(figsize=(12,5))
plt.plot(rewards); plt.title("Episode Rewards"); plt.grid()
plt.savefig(os.path.join(MODEL_DIR, "reward_plot.png"))

plt.figure(figsize=(12,4))
plt.plot(epsilons); plt.title("Epsilon Decay"); plt.grid()
plt.savefig(os.path.join(MODEL_DIR, "epsilon_plot.png"))

if not np.all(np.isnan(losses)):
    plt.figure(figsize=(12,4))
    plt.plot(losses); plt.title("Loss per Episode"); plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, "loss_plot.png"))

print("\n📊 Saved all graphs & training CSV!")
print("\n🎉 RAINBOW TRAINING + TEST COMPLETE!\n")
