# =========================
# Imports & Setup
# =========================

!apt-get install -y swig
!pip install "gymnasium[box2d]" pygame
!pip install gymnasium
!pip install tensorflow

import random
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import Box2D

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Hyperparameter helpers
# =========================
def discount_rate():  # Gamma
    return 1

def learning_rate():  # Alpha
    return 0.001

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
# Double DQN Agent (Keras)
# =========================
class DoubleDeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=10000)   # larger buffer for LunarLander
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Networks
        self.model = self.build_model()          # online network
        self.model_target = self.build_model()   # target network
        self.update_target_from_model()          # sync target
        # Metrics
        self.loss = []

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(self.nS,)))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(self.nA, activation='linear'))
        model.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=self.alpha)
        )
        return model

    def update_target_from_model(self):
        self.model_target.set_weights(self.model.get_weights())

    def action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)
        q_vals = self.model.predict(state.reshape(1, -1), verbose=0)
        return int(np.argmax(q_vals[0]))

    def remember(self, state, action, reward, nstate, done):
        self.memory.append((state, action, reward, nstate, done))

    def experience_replay(self, batch_size):
        # 1) Sample mini-batch
        minibatch = random.sample(self.memory, batch_size)

        # 2) Prepare state arrays
        st = np.zeros((batch_size, self.nS), dtype=np.float32)
        nst = np.zeros((batch_size, self.nS), dtype=np.float32)
        act = np.zeros((batch_size,), dtype=np.int32)
        rew = np.zeros((batch_size,), dtype=np.float32)
        don = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            st[i] = minibatch[i][0].reshape(-1, self.nS)
            nst[i] = minibatch[i][3].reshape(-1, self.nS)
            act[i] = minibatch[i][1]
            rew[i] = minibatch[i][2]
            don[i] = float(minibatch[i][4])

        # 3) Predict Q(s,·) and Q_target(s',·)
        q_st = self.model.predict(st, verbose=0)                 # online
        q_nst_online = self.model.predict(nst, verbose=0)        # for argmax
        q_nst_target = self.model_target.predict(nst, verbose=0) # for value

        # 4) Build DDQN targets
        for i in range(batch_size):
            target = q_st[i]
            if don[i] == 1.0:
                target[act[i]] = rew[i]
            else:
                # action selection from online net
                best_a = np.argmax(q_nst_online[i])
                # action evaluation from target net
                target[act[i]] = rew[i] + self.gamma * q_nst_target[i][best_a]
            q_st[i] = target

        # 5) SGD step
        hist = self.model.fit(st, q_st, epochs=1, verbose=0, batch_size=batch_size)
        self.loss.append(hist.history['loss'][0])

        # 6) Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =========================
# Create Agent & Train
# =========================

# Dimensions
nS = envLunar.observation_space.shape[0]   # 8
nA = envLunar.action_space.n               # 4

# Agent
dqn = DoubleDeepQNetwork(
    nS, nA,
    learning_rate(),
    discount_rate(),
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.910
)

batch = batch_size()

# Training params
EPISODES = 500
MAX_STEPS = 100            # per episode
TARGET_SYNC_EVERY = 1       # episodes
SOLVE_LINE = 200.0          # moving average(100) target for LunarLander

# Tracking
rewards = []
epsilons = []
losses = []

# Training loop
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

    # episode end
    rewards.append(total_reward)
    epsilons.append(dqn.epsilon)
    if len(dqn.loss) > 0:
        losses.append(dqn.loss[-1])
    else:
        losses.append(np.nan)

    if (e + 1) % TARGET_SYNC_EVERY == 0:
        dqn.update_target_from_model()

    # Metrics print
    ma100 = np.mean(rewards[-100:]) if len(rewards) >= 1 else total_reward
    print(f"[Episode {e+1:4d}] reward={total_reward:8.2f}  MA100={ma100:8.2f}  "
          f"epsilon={dqn.epsilon:6.3f}  last_loss={losses[-1]:.6f}  "
          f"memory={len(dqn.memory)}")

    # Early stop if solved (optional)
    if len(rewards) >= 100 and np.mean(rewards[-100:]) >= SOLVE_LINE:
        print("Environment solved based on MA100 threshold. Stopping training...")
        TRAIN_END = e + 1
        break
else:
    TRAIN_END = EPISODES



# =========================
# Testing (no learning)
# =========================
TEST_EPISODES = 10
test_rewards = []
for te in range(TEST_EPISODES):
    s, _ = envLunar.reset(seed=SEED + 1000 + te)
    ep_r = 0.0
    done = False
    while not done:
        q_vals = dqn.model.predict(s.reshape(1, -1), verbose=0)
        a = int(np.argmax(q_vals[0]))
        s, r, terminated, truncated, _ = envLunar.step(a)
        done = terminated or truncated
        ep_r += r
    test_rewards.append(ep_r)

print("==== TEST SUMMARY ====")
print(f"episodes={TEST_EPISODES}")
print(f"mean_reward={np.mean(test_rewards):.2f}  std={np.std(test_rewards):.2f}")
print(f"min_reward={np.min(test_rewards):.2f}  max_reward={np.max(test_rewards):.2f}")

# =========================
# Visualization
# =========================
# Rewards and Moving Average
rolling_average = np.convolve(rewards, np.ones(100)/100, mode='valid') if len(rewards) >= 100 else np.array([])

plt.figure(figsize=(14,7))
plt.plot(rewards, label='Episode Reward')
if rolling_average.size > 0:
    plt.plot(range(99, 99+len(rolling_average)), rolling_average, label='Moving Avg (100)', linewidth=2)
plt.axhline(y=SOLVE_LINE, color='r', linestyle='--', label='Solve Threshold (200)')
plt.xlabel('Episode'); plt.ylabel('Reward'); plt.title('DDQN on LunarLander-v2: Rewards')
plt.legend(); plt.grid(True); plt.show()

# Epsilon
plt.figure(figsize=(14,4))
plt.plot([200*x for x in epsilons], label='Epsilon (scaled x200)')
plt.xlabel('Episode'); plt.ylabel('Scaled Value'); plt.title('Exploration (epsilon) over Episodes')
plt.legend(); plt.grid(True); plt.show()

# Loss
if len(losses) > 0 and not np.all(np.isnan(losses)):
    plt.figure(figsize=(14,4))
    plt.plot(losses, label='Huber/MSE Loss per Episode')
    plt.xlabel('Episode'); plt.ylabel('Loss'); plt.title('Training Loss')
    plt.legend(); plt.grid(True); plt.show()
