import os
import sys
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
# import argparse # REMOVED

# --- Hardcoded Configuration for CI Test ---
ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
VIDEO_DIR = os.path.join(MODEL_DIR, "video")
MODEL_PATH = os.path.join(MODEL_DIR, "ddqn_cartpole_keras.weights.h5")

# Fast Test Configuration
TOTAL_STEPS = 100_000      # Total training timesteps for a quick CI test
STAGE_SIZE = 10_000       # Steps per training stage (for more frequent plotting/evaluation)
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
GAMMA = 0.99
# --- End Configuration ---

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

print(f"TensorFlow version: {tf.__version__}")
print(f"Gymnasium version: {gym.__version__}")

# --- Core Classes ---

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def size(self):
        return len(self.buffer)

class DDQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay_steps=500): # Small decay steps for test
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = (epsilon_start - epsilon_final) / epsilon_decay_steps

        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()

        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.replay_buffer = ReplayBuffer(50_000)
        self.training_step = 0
        self.episode_rewards = []

    def _build_network(self):
        model = keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
            layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
            layers.Dense(self.action_dim, activation='linear')
        ])
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def select_action(self, state, deterministic=False):
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        q_values = self.q_network(np.array([state]), training=False)
        return np.argmax(q_values[0])

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            q_values_actions = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.action_dim), axis=1
            )
            next_q_values = self.q_network(next_states, training=False)
            best_actions = tf.argmax(next_q_values, axis=1, output_type=tf.int32)
            target_q_values = self.target_network(next_states, training=False)
            target_q_values_actions = tf.reduce_sum(
                target_q_values * tf.one_hot(best_actions, self.action_dim), axis=1
            )
            targets = rewards + (1 - tf.cast(dones, tf.float32)) * self.gamma * target_q_values_actions
            loss = tf.reduce_mean(tf.square(targets - q_values_actions))

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss

    def train_one_batch(self, batch_size=32):
        if self.replay_buffer.size() < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        loss = self.train_step(states, actions, rewards, next_states, dones)

        if self.epsilon > self.epsilon_final:
            self.epsilon -= self.epsilon_decay

        self.training_step += 1
        return loss.numpy()

    # Load/save methods kept for completeness, though skipped in this short run
    def load(self, path):
        if os.path.exists(path):
            try:
                self.q_network.load_weights(path)
                self.update_target_network()
                return True
            except:
                return False
        return False

    def save(self, path):
        self.q_network.save_weights(path)


# --- Utility Functions ---

def evaluate_agent(agent, n_episodes=10):
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

def plot_training_progress(reward_progress, stages, losses):
    """Plot training progress: Evaluation Rewards and Training Losses"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Stage Rewards (Mean Evaluation)
    ax1.plot(np.arange(1, stages + 1), reward_progress, marker="o",
             linewidth=2, markersize=8, color='#2E86AB')
    ax1.axhline(y=195, color='orange', linestyle='--', alpha=0.5, label='Success threshold (195)')
    ax1.set_title("DDQN Evaluation Reward Progress", fontsize=14, fontweight='bold')
    ax1.set_xlabel(f"Training Stage (Stage Size: {STAGE_SIZE:,})", fontsize=12)
    ax1.set_ylabel("Mean Evaluation Reward", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')

    # Plot 2: Training Loss (Smoothed)
    if len(losses) > 0:
        window = min(len(losses) // 10, 500)
        
        if len(losses) > 1000:
            ax2.plot(losses, alpha=0.1, linewidth=0.5, color='gray', label='Raw loss')

        if len(losses) >= window:
            smoothed_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax2.plot(range(window - 1, len(losses)), smoothed_loss,
                     linewidth=2, color='#A23B72', label=f'Smoothed (window={window})')

        ax2.set_title("DDQN Training Loss (Mean Squared Error)", fontsize=14, fontweight='bold')
        ax2.set_xlabel(f"Training Step (Total: {len(losses):,})", fontsize=12)
        ax2.set_ylabel("Loss Value", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.set_title("No Loss Data Recorded", fontsize=14, fontweight='bold')

    fig.suptitle(f"CartPole-v1 DDQN Training ({TOTAL_STEPS:,} Steps)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(MODEL_DIR, "training_reward_and_loss_plot_ddqn_cartpole.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved training plot to {plot_path}")
    plt.close(fig) # Close figure to free memory

# --- Main Training Function ---

def train_ddqn(total_steps, stage_size, lr, batch_size, gamma):
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Adjusted epsilon decay steps for the short run
    agent = DDQNAgent(state_dim, action_dim, lr=lr, gamma=gamma,
                      epsilon_decay_steps=total_steps // 10)
    
    stages = max(1, total_steps // stage_size)
    reward_progress = []
    all_losses = []
    
    obs, _ = env.reset(seed=0)
    episode_reward = 0
    global_step = 0
    update_target_every = 250
    train_every = 4

    print(f"\n{'='*60}")
    print(f"Starting training: {stages} stages of {stage_size:,} steps each. Total steps: {total_steps:,}")
    print(f"{'='*60}\n")
    
    try:
        for stage in range(stages):
            print(f"🚀 Stage {stage+1}/{stages} → training {stage_size:,} steps")
            stage_losses = []

            for step in range(stage_size):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.replay_buffer.add(obs, action, reward, next_obs, float(done))
                episode_reward += reward
                global_step += 1

                if global_step % train_every == 0 and agent.replay_buffer.size() > batch_size:
                    loss = agent.train_one_batch(batch_size)
                    if loss > 0:
                        all_losses.append(loss)
                        stage_losses.append(loss)

                if global_step % update_target_every == 0:
                    agent.update_target_network()

                if done:
                    agent.episode_rewards.append(episode_reward)
                    obs, _ = env.reset()
                    episode_reward = 0
                else:
                    obs = next_obs
                
                if global_step >= total_steps:
                    break

            if global_step >= total_steps:
                break

            mean_reward = evaluate_agent(agent, n_episodes=5)
            reward_progress.append(mean_reward)

            avg_loss = np.mean(stage_losses) if stage_losses else 0
            print(f"   📈 Eval reward: {mean_reward:.2f}")
            print(f"   🎯 Epsilon: {agent.epsilon:.3f}")
            print(f"   📊 Avg loss: {avg_loss:.4f}")
            print()

    except Exception as e:
        print(f"CRITICAL ERROR during training: {e}", file=sys.stderr)
        env.close()
        raise

    env.close()
    return agent, reward_progress, all_losses

# --- Main Execution Block (No arguments needed) ---

if __name__ == '__main__':
    print("⚠️ Running a HARDCODED SHORT TEST (5,000 steps).")
    
    agent, reward_progress, training_losses = train_ddqn(
        total_steps=TOTAL_STEPS,
        stage_size=STAGE_SIZE,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA
    )

    print("\n" + "="*60)
    print("✅ Training completed!")
    print("="*60)

    # Plot the results
    stages = len(reward_progress)
    plot_training_progress(reward_progress, stages, training_losses)

    # Final Evaluation Check for CI Pass/Fail
    final_mean_reward = evaluate_agent(agent, n_episodes=10)
    print(f"\nFinal Evaluation (10 episodes): Mean Reward = {final_mean_reward:.2f}")
    
    # Check if the agent learned minimally in 5000 steps (e.g., reached 50 reward)
    MINIMUM_PASS_REWARD = 50
    if final_mean_reward < MINIMUM_PASS_REWARD:
         print(f"❌ CI Test FAILED: Mean reward ({final_mean_reward:.2f}) is below minimum pass threshold ({MINIMUM_PASS_REWARD}).")
         sys.exit(1)
    else:
         print(f"✅ CI Test PASSED: Mean reward ({final_mean_reward:.2f}) meets the threshold.")
