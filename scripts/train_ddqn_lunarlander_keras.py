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
# Device Selection (TF CPU/CUDA/Metal)
# =========================
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    DEVICE = "/GPU:0"   # Works on CUDA + Apple Metal (tf-macos)
else:
    DEVICE = "/CPU:0"

print(f"🔧 TensorFlow executing on device: {DEVICE}")

# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Hyperparameters
# =========================
def discount_rate():  # Gamma
    return 1

def batch_size():
    return 64

# =========================
# Environment
# =========================
envLunar = gym.make('LunarLander-v3')
envLunar.reset(seed=SEED)
envLunar.action_space.seed(SEED)
envLunar.observation_space.seed(SEED)

# =========================
# Double Deep Q-Network (Keras)
# =========================
class DoubleDeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=10000)
        self.alpha = alpha
        self.gamma = gamma

        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Networks
        with tf.device(DEVICE):
            self.model = self.build_model()
            self.model_target = self.build_model()
            self.update_target_from_model()

        # Metrics
        self.loss = []

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.nS,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.nA, activation='linear'),
        ])
        model.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=self.alpha)
        )
        return model

    def update_target_from_model(self):
        self.model_target.set_weights(self.model.get_weights())

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)
        with tf.device(DEVICE):
            q_vals = self.model.predict(state.reshape(1, -1), verbose=0)
        return int(np.argmax(q_vals[0]))

    def remember(self, state, action, reward, nstate, done):
        self.memory.append((state, action, reward, nstate, done))

    def experience_replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        st = np.zeros((batch_size, self.nS), dtype=np.float32)
        nst = np.zeros((batch_size, self.nS), dtype=np.float32)
        act = np.zeros((batch_size,), dtype=np.int32)
        rew = np.zeros((batch_size,), dtype=np.float32)
        don = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            st[i] = minibatch[i][0].reshape(-1, self.nS)
            nst[i] = minibatch[i][3].reshape(-1, self.nS)
            act[i], rew[i], don[i] = minibatch[i][1], minibatch[i][2], float(minibatch[i][4])

        with tf.device(DEVICE):
            q_st = self.model.predict(st, verbose=0)
            q_nst_online = self.model.predict(nst, verbose=0)
            q_nst_target = self.model_target.predict(nst, verbose=0)

        for i in range(batch_size):
            target = q_st[i]
            if don[i] == 1.0:
                target[act[i]] = rew[i]
            else:
                best_a = np.argmax(q_nst_online[i])
                target[act[i]] = rew[i] + self.gamma * q_nst_target[i][best_a]
            q_st[i] = target

        with tf.device(DEVICE):
            hist = self.model.fit(st, q_st, epochs=1, verbose=0, batch_size=batch_size)

        self.loss.append(hist.history['loss'][0])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =========================
# Create Agent & Train
# =========================

# Model Folder
MODEL_DIR = "keras_ddqn_models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "ddqn_lunarlander_v3.h5")

# CLI Args
parser = argparse.ArgumentParser(description="Train DDQN on LunarLander-v3 (Keras)")
parser.add_argument("--total_steps", type=int, default=500)  # episodes
parser.add_argument("--stage_size", type=int, default=500)   # MAX STEPS PER EPISODE
parser.add_argument("--lr", type=float, default=5e-4)
args = parser.parse_args()

EPISODES = args.total_steps
MAX_STEPS = args.stage_size
LR = args.lr
TARGET_SYNC_EVERY = 1
SOLVE_LINE = 200.0

# Dimensions
nS, nA = envLunar.observation_space.shape[0], envLunar.action_space.n

# Agent
dqn = DoubleDeepQNetwork(nS, nA, LR, discount_rate(), epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.910)

batch = batch_size()
rewards, epsilons, losses = [], [], []

for e in range(EPISODES):
    state, _ = envLunar.reset(seed=SEED + e)
    total_reward = 0.0

    for t in range(MAX_STEPS):
        action = dqn.action(state)
        nstate, reward, terminated, truncated, _ = envLunar.step(action)
        done = terminated or truncated

        dqn.remember(state, action, reward, nstate, done)
        if len(dqn.memory) > batch:
            dqn.experience_replay(batch)

        state = nstate
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)
    epsilons.append(dqn.epsilon)
    losses.append(dqn.loss[-1] if len(dqn.loss) > 0 else np.nan)

    if (e + 1) % TARGET_SYNC_EVERY == 0:
        dqn.update_target_from_model()

    print(f"[Episode {e+1}] reward={total_reward:.2f} epsilon={dqn.epsilon:.3f} last_loss={losses[-1]:.6f}")

# Save Model
dqn.model.save(MODEL_PATH)
print(f"💾 Saved model → {MODEL_PATH}")

# =========================
# TEST SUMMARY (NO LEARNING)
# =========================
TEST_EPISODES = 20
test_rewards = []

for te in range(TEST_EPISODES):
    s, _ = envLunar.reset(seed=SEED + 1000 + te)
    ep_r = 0.0
    done = False
    while not done:
        with tf.device(DEVICE):
            q_vals = dqn.model.predict(s.reshape(1, -1), verbose=0)
        a = int(np.argmax(q_vals[0]))
        s, r, terminated, truncated, _ = envLunar.step(a)
        done = terminated or truncated
        ep_r += r
    test_rewards.append(ep_r)

print("\n==== 🧪 TEST SUMMARY ====")
print(f"episodes={TEST_EPISODES}")
print(f"mean_reward={np.mean(test_rewards):.2f}  std={np.std(test_rewards):.2f}")
print(f"min_reward={np.min(test_rewards):.2f}  max_reward={np.max(test_rewards):.2f}")

# =========================
# SAVE Graphs + CSV
# =========================
results_df = pd.DataFrame({"reward": rewards, "epsilon": epsilons, "loss": losses})
results_df.to_csv(os.path.join(MODEL_DIR, "training_data.csv"), index=False)

plt.figure(figsize=(12,5))
plt.plot(rewards); plt.title("Rewards per Episode"); plt.grid()
plt.savefig(os.path.join(MODEL_DIR, "reward_plot.png"))

plt.figure(figsize=(12,4))
plt.plot(epsilons); plt.title("Epsilon Decay"); plt.grid()
plt.savefig(os.path.join(MODEL_DIR, "epsilon_plot.png"))

if not np.all(np.isnan(losses)):
    plt.figure(figsize=(12,4))
    plt.plot(losses); plt.title("Training Loss per Episode"); plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, "loss_plot.png"))

print("📊 Saved all graphs + CSV!")
print("\n🎉 TRAINING + TEST COMPLETE!\n")
