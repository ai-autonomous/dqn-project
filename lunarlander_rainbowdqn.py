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
    return 0.005

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
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

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
# Rainbow-style DQN Agent
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

        self.model = self.build_model()
        self.model_target = self.build_model()
        self.update_target_from_model()

        self.loss = []

    def build_model(self):
        inputs = keras.layers.Input(shape=(self.nS,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dense(128, activation='relu')(x)

        V = keras.layers.Dense(1, activation=None)(x)
        A = keras.layers.Dense(self.nA, activation=None)(x)

        #A_mean = keras.layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(A)
        A_mean = keras.layers.Lambda(
                    lambda a: tf.reduce_mean(a, axis=1, keepdims=True),
                    output_shape=(1,))(A)
        Q = keras.layers.Add()([V, keras.layers.Subtract()([A, A_mean])])

        model = keras.Model(inputs=inputs, outputs=Q)
        model.compile(
            loss='mean_squared_error',
            optimizer=keras.optimizers.Adam(learning_rate=self.alpha)
        )
        return model

    def update_target_from_model(self):
        self.model_target.set_weights(self.model.get_weights())

    def _get_beta(self):
        return min(1.0, self.per_beta_start + (1.0 - self.per_beta_start) * (self.frame_idx / float(self.per_beta_frames)))

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)
        q_vals = self.model.predict(state.reshape(1, -1), verbose=0)
        return int(np.argmax(q_vals[0]))

    def _store_n_step(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return None
        R = 0.0
        for idx, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
        state, action, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state, done = self.n_step_buffer[-1]
        return (state, action, R, next_state, done)

    def remember(self, state, action, reward, nstate, done):
        n_step_transition = self._store_n_step((state, action, reward, nstate, done))
        if n_step_transition:
            s, a, R, ns, d = n_step_transition
            self.memory.push(s, a, R, ns, d)

    def experience_replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        beta = self._get_beta()
        self.frame_idx += 1

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, beta=beta)

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

        weights = weights.reshape(-1, 1)
        hist = self.model.fit(states, q_st, epochs=1, verbose=0, batch_size=batch_size, sample_weight=weights.squeeze())
        self.loss.append(hist.history['loss'][0])

        new_priorities = np.abs(td_errors) + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =========================
# Create Rainbow Agent & Train
# =========================
nS = envLunar.observation_space.shape[0]
nA = envLunar.action_space.n

rainbow = RainbowDQN(
    nS, nA,
    learning_rate(),
    discount_rate(),
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    buffer_size=50000,
    n_step=3,
    per_alpha=0.6,
    per_beta_start=0.4,
    per_beta_frames=100000
)

batch = batch_size()

EPISODES = 100
MAX_STEPS = 50
TARGET_SYNC_EVERY = 5
SOLVE_LINE = 200.0

rewards = []
epsilons = []
losses = []

for e in range(EPISODES):
    state, _ = envLunar.reset(seed=SEED + e)
    total_reward = 0.0

    for t in range(MAX_STEPS):
        action = rainbow.action(state)
        nstate, reward, terminated, truncated, _ = envLunar.step(action)
        done = terminated or truncated

        rainbow.remember(state, action, reward, nstate, done)
        if len(rainbow.memory) > batch:
            rainbow.experience_replay(batch)

        state = nstate
        total_reward += reward
        if done:
            break

    rewards.append(total_reward)
    epsilons.append(rainbow.epsilon)
    if len(rainbow.loss) > 0:
        losses.append(rainbow.loss[-1])
    else:
        losses.append(np.nan)

    if (e + 1) % TARGET_SYNC_EVERY == 0:
        rainbow.update_target_from_model()

    ma100 = np.mean(rewards[-100:]) if len(rewards) >= 1 else total_reward
    print(f"[Episode {e+1:4d}] reward={total_reward:8.2f}  MA100={ma100:8.2f}  "
          f"epsilon={rainbow.epsilon:6.3f}  last_loss={losses[-1]:.6f}  memory={len(rainbow.memory)}")

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
        q_vals = rainbow.model.predict(s.reshape(1, -1), verbose=0)
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
rolling_average = np.convolve(rewards, np.ones(100)/100, mode='valid') if len(rewards) >= 100 else np.array([])

plt.figure(figsize=(14,7))
plt.plot(rewards, label='Episode Reward')
if rolling_average.size > 0:
    plt.plot(range(99, 99+len(rolling_average)), rolling_average, label='Moving Avg (100)', linewidth=2)
plt.axhline(y=200.0, color='r', linestyle='--', label='Solve Threshold (200)')
plt.xlabel('Episode'); plt.ylabel('Reward'); plt.title('Rainbow-style DQN on LunarLander-v3: Rewards')
plt.legend(); plt.grid(True); plt.show()

plt.figure(figsize=(14,4))
plt.plot([200*x for x in epsilons], label='Epsilon (scaled x200)')
plt.xlabel('Episode'); plt.ylabel('Scaled Value'); plt.title('Exploration (epsilon) over Episodes')
plt.legend(); plt.grid(True); plt.show()

if len(losses) > 0 and not np.all(np.isnan(losses)):
    plt.figure(figsize=(14,4))
    plt.plot(losses, label='Loss per Episode')
    plt.xlabel('Episode'); plt.ylabel('Loss'); plt.title('Training Loss (Rainbow-style DQN)')
    plt.legend(); plt.grid(True); plt.show()
