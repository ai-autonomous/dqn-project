import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from collections import deque
import random
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output, display, HTML
import base64
import shutil # For saving to Google Drive

warnings.filterwarnings("ignore")

print(f"TensorFlow version: {tf.__version__}")
print(f"Gymnasium version: {gym.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# --- Configuration ---
# Training Configuration
TOTAL_STEPS = 100_000      # Total training timesteps
STAGE_SIZE = 10_000        # Steps per training stage (for evaluation/saving)
LEARNING_RATE = 5e-4       # Learning rate
BATCH_SIZE = 32            # Batch size for training
GAMMA = 0.99               # Discount factor

# Environment
ENV_NAME = "CartPole-v1"

print(f"🎯 Dueling DDQN Training Configuration:")
print(f"   Environment: {ENV_NAME}")
print(f"   Total Steps: {TOTAL_STEPS:,}")
print(f"   Stage Size: {STAGE_SIZE:,}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Gamma: {GAMMA}")

# --- Replay Buffer ---

class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a random batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def size(self):
        """Return current buffer size"""
        return len(self.buffer)

print("✅ ReplayBuffer class defined")

# --- Dueling DDQN Agent (Final Fix: Enforcing Python int for Lambda) ---

class DDQNAgent:
    """Double Deep Q-Network Agent with Dueling Architecture"""

    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay_steps=10_000):
        self.state_dim = state_dim
        # FIX: Ensure action_dim is a standard Python int, not np.int64
        self.action_dim = int(action_dim)
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = (epsilon_start - epsilon_final) / epsilon_decay_steps

        # Build networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()

        self.optimizer = Adam(learning_rate=lr)
        self.replay_buffer = ReplayBuffer(50_000)

        # Training stats
        self.training_step = 0
        self.episode_rewards = []

    def _build_network(self):
        """Builds the Dueling Q-network"""
        input_layer = Input(shape=(self.state_dim,))

        # Common feature layers
        x = Dense(128, activation='relu', kernel_initializer='he_uniform')(input_layer)
        x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)

        # --- Value Stream V(s) ---
        value_stream = Dense(1, kernel_initializer='he_uniform')(x)

        # --- Advantage Stream A(s, a) ---
        advantage_stream = Dense(self.action_dim, kernel_initializer='he_uniform')(x)

        # --- Combine V(s) and A(s, a) to get Q(s, a) ---
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = Lambda(
            lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
            # Use the simple tuple notation, which should now work since self.action_dim is an int.
            output_shape=(self.action_dim,)
        )([value_stream, advantage_stream])
        #

        # Create the model using the Keras Functional API
        model = Model(inputs=input_layer, outputs=q_values)
        return model

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())

    def select_action(self, state, deterministic=False):
        """Epsilon-greedy action selection"""
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        q_values = self.q_network(np.array([state]), training=False)
        return np.argmax(q_values[0])

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform one training step using Double DQN (wrapped in @tf.function for GPU speed)"""

        with tf.GradientTape() as tape:
            # Current Q-values
            q_values = self.q_network(states, training=True)
            q_values_actions = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.action_dim), axis=1
            )

            # Double DQN:
            next_q_values = self.q_network(next_states, training=False)
            best_actions = tf.argmax(next_q_values, axis=1, output_type=tf.int32)

            target_q_values = self.target_network(next_states, training=False)
            target_q_values_actions = tf.reduce_sum(
                target_q_values * tf.one_hot(best_actions, self.action_dim), axis=1
            )

            targets = rewards + (1 - tf.cast(dones, tf.float32)) * self.gamma * target_q_values_actions

            # Loss (Mean Squared Error)
            loss = tf.reduce_mean(tf.square(targets - q_values_actions))

        # Update Q-network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        return loss

    def train_step_wrapper(self, batch_size=32):
        """Wrapper to handle sampling and calling the tf.function train_step"""
        if self.replay_buffer.size() < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to Tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)

        loss = self.train_step(states, actions, rewards, next_states, dones).numpy()

        # Decay epsilon
        if self.epsilon > self.epsilon_final:
            self.epsilon -= self.epsilon_decay

        self.training_step += 1
        return loss

    def save(self, path):
        """Save model weights"""
        self.q_network.save_weights(path)

    def load(self, path):
        """Load model weights"""
        if os.path.exists(path):
            try:
                # Need to build the model first to load weights
                dummy_state = np.zeros((1, self.state_dim))
                self.q_network(dummy_state)
                self.q_network.load_weights(path)
                self.update_target_network()
                return True
            except Exception as e:
                print(f"Error loading model weights: {e}")
                return False
        return False

print("✅ Dueling DDQNAgent class defined (Final Fix: Type Enforced)")

# --- Utility Functions (Unchanged) ---

def termination_reason(terminated, truncated, steps, max_steps):
    """Classify episode termination reason"""
    if truncated and steps == max_steps:
        # CartPole-v1 is solved if 500 steps (max_steps) are reached.
        return "SOLVED"
    elif truncated:
        return "TIME_LIMIT"
    elif terminated and steps >= 195:
        # CartPole-v0 success threshold is 195, good to track
        return "GOOD_RUN"
    elif terminated:
        return "FAIL"
    else:
        return "UNKNOWN"

def evaluate_agent(agent, n_episodes=10):
    """Evaluate agent performance (non-training mode)"""
    env = gym.make(ENV_NAME)
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=100 + ep)
        episode_reward = 0

        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                rewards.append(episode_reward)
                break

    env.close()
    return np.mean(rewards)

best_reward = -999999

print("✅ Utility functions defined")

# --- Training Function (Modified to return losses) ---

def train_ddqn(total_steps, stage_size, lr, batch_size, gamma):
    """Main training loop - returns agent, reward_progress, and all losses"""
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Decay epsilon over the first 10% of total steps
    agent = DDQNAgent(state_dim, action_dim, lr=lr, gamma=gamma,
                      epsilon_decay_steps=int(total_steps * 0.1))

    stages = max(1, total_steps // stage_size)
    reward_progress = [] # List of mean evaluation rewards per stage
    all_losses = []      # List of all training losses

    obs, _ = env.reset(seed=0)
    episode_reward = 0
    episode_steps = 0
    global_step = 0
    update_target_every = 250
    train_every = 4 # Train the agent every 4 steps

    print(f"\n{'='*60}")
    print(f"Starting Dueling DDQN training: {stages} stages of {stage_size:,} steps each")
    print(f"Total steps: {total_steps:,}")
    print(f"{'='*60}\n")

    for stage in range(stages):
        print(f"🚀 Stage {stage+1}/{stages} (Steps {global_step+1:,} to {(stage+1)*stage_size:,})")
        stage_losses = []

        for step in range(stage_size):
            # Select and perform action
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.replay_buffer.add(obs, action, reward, next_obs, float(done))

            episode_reward += reward
            episode_steps += 1
            global_step += 1

            # Train agent
            if global_step % train_every == 0 and agent.replay_buffer.size() > batch_size:
                # Use the new wrapper function
                loss = agent.train_step_wrapper(batch_size)
                if loss > 0:
                    all_losses.append(loss)
                    stage_losses.append(loss)

            # Update target network
            if global_step % update_target_every == 0:
                agent.update_target_network()

            # Episode end
            if done:
                agent.episode_rewards.append(episode_reward)
                obs, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
            else:
                obs = next_obs

            if global_step >= total_steps:
                break

        if global_step >= total_steps:
                break

        # Stage evaluation
        mean_reward = evaluate_agent(agent, n_episodes=10)
        reward_progress.append(mean_reward)

        avg_loss = np.mean(stage_losses) if stage_losses else 0
        print(f"   📈 Eval reward: {mean_reward:.2f}")
        print(f"   🎯 Epsilon: {agent.epsilon:.3f}")
        print(f"   💾 Buffer size: {agent.replay_buffer.size():,}")
        print(f"   📊 Avg loss (last {stage_size:,} steps): {avg_loss:.4f}")
        print(f"   🏆 Episodes completed: {len(agent.episode_rewards)}")

    env.close()

    # RETURN THE LIST OF ALL RECORDED LOSSES
    return agent, reward_progress, all_losses

print("✅ Training function defined")

# --- Plotting Function (Unchanged) ---

def plot_training_progress(reward_progress, stages, episode_rewards, losses):
    """Plot training progress: Evaluation Rewards and Training Losses"""
    # Create two subplots for rewards and loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Plot 1: Stage Rewards (Mean Evaluation) ---
    ax1.plot(np.arange(1, stages + 1), reward_progress, marker="o",
             linewidth=2, markersize=8, color='#2E86AB')
    # Success is defined as achieving an average reward of 475 over 100 consecutive trials,
    # but 195 is the v0 threshold and a good intermediate target.
    ax1.axhline(y=195, color='orange', linestyle='--', alpha=0.5, label='Success threshold (195)')
    ax1.axhline(y=500, color='green', linestyle='--', alpha=0.7, label='Max episode steps (500)')
    ax1.set_title("Dueling DDQN Evaluation Reward Progress", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Training Stage", fontsize=12)
    ax1.set_ylabel("Mean Evaluation Reward", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # --- Plot 2: Training Loss (Smoothed) ---
    if len(losses) > 0:
        ax2.plot(losses, alpha=0.1, linewidth=0.5, color='gray', label='Raw loss')

        # Smoothing window for loss, typically larger than for episode rewards
        window = min(len(losses) // 10, 1000)
        if len(losses) >= window:
            smoothed_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
            # The indices for the smoothed plot start at (window - 1)
            ax2.plot(range(window - 1, len(losses)), smoothed_loss,
                     linewidth=2, color='#A23B72', label=f'Smoothed (window={window})')

        ax2.set_title("Dueling DDQN Training Loss (Mean Squared Error)", fontsize=14, fontweight='bold')
        ax2.set_xlabel(f"Training Step (Total Loss Records: {len(losses):,})", fontsize=12)
        ax2.set_ylabel("Loss Value", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.set_title("No Loss Data Recorded", fontsize=14, fontweight='bold')


    plt.tight_layout()
    plt.show()

print("✅ Plotting functions defined")

# --- Comprehensive Evaluation (Unchanged) ---

def evaluate_and_report(agent, n_eval_episodes=20):
    """Comprehensive evaluation with termination analysis"""
    env = gym.make(ENV_NAME)
    max_steps = env.spec.max_episode_steps

    results = {"SOLVED": 0, "GOOD_RUN": 0, "FAIL": 0, "TIME_LIMIT": 0, "UNKNOWN": 0}
    rewards = []

    print(f"\n{'='*60}")
    print(f"Running final evaluation ({n_eval_episodes} episodes)...")
    print(f"{'='*60}\n")

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=999 + ep)
        ep_reward, steps = 0.0, 0

        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            steps += 1

            if terminated or truncated:
                outcome = termination_reason(terminated, truncated, steps, max_steps)
                results[outcome] += 1
                rewards.append(ep_reward)
                break

    env.close()

    print("🧪 Evaluation Summary:")
    print(f"{'─'*40}")
    for k, v in results.items():
        percentage = (v / n_eval_episodes) * 100
        print(f"   {k:>12}: {v:2d} ({percentage:5.1f}%)")
    print(f"{'─'*40}")
    print(f"   Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"   Min Reward:  {np.min(rewards):.2f}")
    print(f"   Max Reward:  {np.max(rewards):.2f}")
    print(f"{'─'*40}\n")

print("✅ Evaluation function defined")

# --- Main Execution ---

# Train the agent (Catching the new 'training_losses' return value)
agent, reward_progress, training_losses = train_ddqn(
    total_steps=TOTAL_STEPS,
    stage_size=STAGE_SIZE,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA
)

print("\n" + "="*60)
print("✅ Dueling DDQN Training completed!")
print("="*60)

# Plot the results
stages = len(reward_progress)
# Pass the list of all losses to the plotting function
plot_training_progress(reward_progress, stages, agent.episode_rewards, training_losses)

# Comprehensive evaluation
evaluate_and_report(agent, n_eval_episodes=20)

except Exception as e:
    print(f"⚠️ Could not mount Drive or save model: {e}")
print("="*60)
