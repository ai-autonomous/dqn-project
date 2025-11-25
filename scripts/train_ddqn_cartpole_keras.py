"""
Double DQN implementation for CartPole-v1 using Keras/TensorFlow.
Improved version with stability enhancements for consistent training across environments.

Key improvements:
  • Better seeding for reproducibility
  • Increased training stability
  • Warm-up period before training
  • Gradient clipping
  • More robust hyperparameters
  • Better handling of terminal states
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/CD
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

# ===================== CONFIGURATION =====================
def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Double DQN on CartPole-v1 with Keras",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--total_steps", 
        type=int, 
        default=200_000,  # Increased for more stable training
        help="Total training timesteps"
    )
    parser.add_argument(
        "--stage_size", 
        type=int, 
        default=20_000,  # Increased for better evaluation
        help="Steps per training stage"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=2.5e-4,  # Reduced for more stable learning
        help="Learning rate"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64,  # Increased for more stable gradients
        help="Batch size for training"
    )
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.99, 
        help="Discount factor"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model_dir", 
        type=str, 
        default="models", 
        help="Directory to save models and outputs"
    )
    parser.add_argument(
        "--no_video", 
        action="store_true", 
        help="Disable video recording"
    )
    parser.add_argument(
        "--eval_episodes", 
        type=int, 
        default=20, 
        help="Number of episodes for final evaluation"
    )
    parser.add_argument(
        "--buffer_size", 
        type=int, 
        default=100_000,  # Increased to prevent forgetting
        help="Replay buffer size"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=1000,  # Warmup before training
        help="Steps to collect before training starts"
    )
    
    return parser.parse_args()


# ===================== REPLAY BUFFER =====================
class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""
    
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a random batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32), 
                np.array(actions, dtype=np.int32), 
                np.array(rewards, dtype=np.float32), 
                np.array(next_states, dtype=np.float32), 
                np.array(dones, dtype=np.float32))
    
    def size(self):
        """Return current buffer size"""
        return len(self.buffer)


# ===================== DDQN AGENT =====================
class DDQNAgent:
    """Double Deep Q-Network Agent with stability improvements"""
    
    def __init__(self, state_dim, action_dim, lr=2.5e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay_steps=20_000,
                 buffer_size=100_000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = (epsilon_start - epsilon_final) / epsilon_decay_steps
        
        # Build networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Optimizer with gradient clipping
        self.optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=10.0)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        
    def _build_network(self):
        """Build Q-network with improved architecture"""
        model = keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation='relu', 
                        kernel_initializer=keras.initializers.HeNormal()),
            layers.Dense(128, activation='relu', 
                        kernel_initializer=keras.initializers.HeNormal()),
            layers.Dense(self.action_dim, activation='linear',
                        kernel_initializer=keras.initializers.HeNormal())
        ])
        return model
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def select_action(self, state, deterministic=False):
        """Epsilon-greedy action selection"""
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = np.array([state], dtype=np.float32)
        q_values = self.q_network(state, training=False)
        return int(np.argmax(q_values[0]))
    
    @tf.function
    def _train_step_graph(self, states, actions, rewards, next_states, dones):
        """Compiled training step for efficiency"""
        with tf.GradientTape() as tape:
            # Current Q-values
            q_values = self.q_network(states, training=True)
            one_hot_actions = tf.one_hot(actions, self.action_dim)
            q_values_actions = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            
            # Double DQN: use Q-network to select, target to evaluate
            next_q_values = self.q_network(next_states, training=False)
            best_actions = tf.argmax(next_q_values, axis=1, output_type=tf.int32)
            
            target_q_values = self.target_network(next_states, training=False)
            one_hot_best = tf.one_hot(best_actions, self.action_dim)
            target_q_values_actions = tf.reduce_sum(
                target_q_values * one_hot_best, axis=1
            )
            
            # Compute targets with proper terminal handling
            targets = rewards + (1.0 - dones) * self.gamma * target_q_values_actions
            
            # Huber loss for robustness
            loss = tf.reduce_mean(tf.keras.losses.huber(targets, q_values_actions))
        
        # Update Q-network with gradient clipping
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.q_network.trainable_variables)
        )
        
        return loss
    
    def train_step(self, batch_size=32):
        """Perform one training step using Double DQN"""
        if self.replay_buffer.size() < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        loss = self._train_step_graph(
            tf.constant(states),
            tf.constant(actions),
            tf.constant(rewards),
            tf.constant(next_states),
            tf.constant(dones)
        )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_final:
            self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_decay)
        
        self.training_step += 1
        return float(loss)
    
    def save(self, path):
        """Save model weights"""
        self.q_network.save_weights(path)
    
    def load(self, path):
        """Load model weights"""
        if os.path.exists(path):
            self.q_network.load_weights(path)
            self.update_target_network()
            return True
        return False


# ===================== UTILITY FUNCTIONS =====================
def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def termination_reason(terminated, truncated, steps, max_steps):
    """Classify episode termination reason"""
    if truncated and steps >= max_steps:
        return "SOLVED"
    elif truncated:
        return "TIME_LIMIT"
    elif terminated and steps >= 195:
        return "GOOD_RUN"
    elif terminated:
        return "FAIL"
    else:
        return "UNKNOWN"


def evaluate_agent(agent, env_name, n_episodes=10, seed_offset=100):
    """Evaluate agent performance"""
    env = gym.make(env_name)
    rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        episode_reward = 0
        
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                rewards.append(episode_reward)
                break
    
    env.close()
    return np.mean(rewards), np.std(rewards)


# ===================== VIDEO RECORDING =====================
best_reward = -999999

def record_best_video(agent, env_name, video_dir):
    """Record video of best episode only"""
    global best_reward
    
    try:
        env = gym.make(env_name, render_mode="rgb_array")
        env = RecordVideo(env, video_dir, name_prefix="best_cartpole_episode")
        
        obs, _ = env.reset(seed=777)
        total_reward, steps = 0, 0
        
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Only keep video if it's a new best
        if total_reward > best_reward:
            best_reward = total_reward
            print(f"🎥 New best score! Saved video: reward={total_reward:.2f}, steps={steps}")
            return True
        else:
            # Remove non-best video
            for f in os.listdir(video_dir):
                if "best_cartpole_episode" in f:
                    try:
                        os.remove(os.path.join(video_dir, f))
                    except:
                        pass
            return False
    except Exception as e:
        print(f"⚠️  Video recording failed: {e}")
        return False


# ===================== TRAINING =====================
def train_ddqn(env_name, total_steps, stage_size, lr, batch_size, gamma, 
               model_path, video_dir, enable_video=True, seed=42, 
               buffer_size=100_000, warmup_steps=1000):
    """Main training loop with stability improvements"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Calculate epsilon decay over longer period for stability
    epsilon_decay_steps = min(int(total_steps * 0.2), 50_000)
    
    agent = DDQNAgent(
        state_dim, action_dim, lr=lr, gamma=gamma, 
        epsilon_decay_steps=epsilon_decay_steps,
        buffer_size=buffer_size
    )
    
    # Load existing model if available
    if agent.load(model_path):
        print(f"📦 Loaded existing model from {model_path}")
    else:
        print(f"🆕 Created new DDQN agent")
    
    stages = max(1, total_steps // stage_size)
    reward_progress = []
    std_progress = []
    
    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_count = 0
    global_step = 0
    update_target_every = 500  # More frequent updates
    train_every = 4
    
    print(f"\n{'='*70}")
    print(f"Starting training: {stages} stages of {stage_size:,} steps each")
    print(f"Warmup steps: {warmup_steps:,}, Epsilon decay: {epsilon_decay_steps:,}")
    print(f"{'='*70}\n")
    
    # Warmup phase
    print(f"🔄 Warming up replay buffer... (target: {warmup_steps:,} steps)")
    while agent.replay_buffer.size() < warmup_steps:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.replay_buffer.add(obs, action, reward, next_obs, float(done))
        
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
    
    print(f"✅ Warmup complete! Buffer size: {agent.replay_buffer.size():,}\n")
    
    # Reset environment after warmup
    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    
    for stage in range(stages):
        print(f"🚀 Stage {stage+1}/{stages}")
        stage_losses = []
        stage_start_episodes = len(agent.episode_rewards)
        
        for step in range(stage_size):
            # Select and perform action
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.replay_buffer.add(obs, action, reward, next_obs, float(done))
            
            episode_reward += reward
            global_step += 1
            
            # Train agent
            if global_step % train_every == 0:
                loss = agent.train_step(batch_size)
                if loss > 0:
                    agent.losses.append(loss)
                    stage_losses.append(loss)
            
            # Update target network
            if global_step % update_target_every == 0:
                agent.update_target_network()
            
            # Episode end
            if done:
                agent.episode_rewards.append(episode_reward)
                episode_count += 1
                obs, _ = env.reset()
                episode_reward = 0
            else:
                obs = next_obs
        
        # Stage evaluation
        mean_reward, std_reward = evaluate_agent(agent, env_name, n_episodes=20)
        reward_progress.append(mean_reward)
        std_progress.append(std_reward)
        
        stage_episodes = len(agent.episode_rewards) - stage_start_episodes
        avg_loss = np.mean(stage_losses) if stage_losses else 0
        recent_rewards = agent.episode_rewards[-50:] if len(agent.episode_rewards) >= 50 else agent.episode_rewards
        avg_train_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        print(f"   📈 Eval: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   🎯 Train (last 50): {avg_train_reward:.2f}")
        print(f"   🎲 Epsilon: {agent.epsilon:.3f}")
        print(f"   💾 Buffer: {agent.replay_buffer.size():,}")
        print(f"   📊 Loss: {avg_loss:.4f}")
        print(f"   🏆 Episodes: {stage_episodes} (total: {len(agent.episode_rewards)})")
        
        # Check if solved
        if mean_reward >= 195.0:
            print(f"   ✅ SOLVED! Mean reward {mean_reward:.2f} >= 195.0")
        
        print()
        
        # Save model
        agent.save(model_path)
        
        # Record best video
        if enable_video and stage % 2 == 0:  # Every other stage to save time
            record_best_video(agent, env_name, video_dir)
    
    env.close()
    
    return agent, reward_progress, std_progress


# ===================== EVALUATION =====================
def evaluate_and_report(agent, env_name, n_eval_episodes=20, seed_offset=999):
    """Comprehensive evaluation with termination analysis"""
    env = gym.make(env_name)
    max_steps = env.spec.max_episode_steps
    
    results = {"SOLVED": 0, "GOOD_RUN": 0, "FAIL": 0, "TIME_LIMIT": 0, "UNKNOWN": 0}
    rewards = []
    
    print(f"\n{'='*70}")
    print(f"Final Evaluation ({n_eval_episodes} episodes)")
    print(f"{'='*70}\n")
    
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
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
    print(f"{'─'*50}")
    for k, v in results.items():
        percentage = (v / n_eval_episodes) * 100
        print(f"   {k:>12}: {v:2d} ({percentage:5.1f}%)")
    print(f"{'─'*50}")
    print(f"   Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"   Min Reward:  {np.min(rewards):.2f}")
    print(f"   Max Reward:  {np.max(rewards):.2f}")
    print(f"{'─'*50}\n")
    
    return results, rewards


# ===================== PLOTTING =====================
def plot_training_progress(reward_progress, std_progress, stages, episode_rewards, output_dir):
    """Plot and save training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Stage rewards with error bars
    stages_x = np.arange(1, stages + 1)
    ax = axes[0, 0]
    ax.errorbar(stages_x, reward_progress, yerr=std_progress, 
                 marker="o", linewidth=2, markersize=8, capsize=5, 
                 color='#2E86AB', ecolor='lightblue', label='Mean ± Std')
    ax.axhline(y=195, color='green', linestyle='--', alpha=0.7, 
               linewidth=2, label='Success threshold (195)')
    ax.set_title("DDQN Training Progress on CartPole-v1", fontsize=14, fontweight='bold')
    ax.set_xlabel("Training Stage", fontsize=12)
    ax.set_ylabel("Mean Evaluation Reward", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Episode rewards (smoothed)
    ax = axes[0, 1]
    if len(episode_rewards) > 0:
        ax.plot(episode_rewards, alpha=0.3, linewidth=0.5, color='gray', label='Raw rewards')
        
        window = min(100, max(10, len(episode_rewards) // 10))
        if len(episode_rewards) >= window:
            smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(episode_rewards)), smoothed, 
                    linewidth=2, color='#A23B72', label=f'Smoothed (window={window})')
        
        ax.axhline(y=195, color='green', linestyle='--', alpha=0.7, 
                   linewidth=2, label='Success threshold')
        ax.set_title("Episode Rewards During Training", fontsize=14, fontweight='bold')
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Episode Reward", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Success rate over time
    ax = axes[1, 0]
    if len(episode_rewards) >= 100:
        success_rate = []
        window = 100
        for i in range(len(episode_rewards) - window + 1):
            rate = np.sum(np.array(episode_rewards[i:i+window]) >= 195) / window * 100
            success_rate.append(rate)
        ax.plot(range(window-1, len(episode_rewards)), success_rate, 
                linewidth=2, color='#F18F01')
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.7, 
                   linewidth=2, label='90% success (solved)')
        ax.set_title("Success Rate (100-episode window)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
    
    # Distribution of final rewards
    ax = axes[1, 1]
    if len(reward_progress) > 0:
        ax.bar(stages_x, reward_progress, color='#2E86AB', alpha=0.7, 
               edgecolor='black', linewidth=1)
        ax.axhline(y=195, color='green', linestyle='--', alpha=0.7, 
                   linewidth=2, label='Success threshold')
        ax.set_title("Evaluation Reward by Stage", fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Stage", fontsize=12)
        ax.set_ylabel("Mean Evaluation Reward", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_reward_plot_ddqn_cartpole.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved training plot to {plot_path}")
    plt.close()


# ===================== MAIN =====================
def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup directories
    ENV_NAME = "CartPole-v1"
    VIDEO_DIR = os.path.join(args.model_dir, "video")
    MODEL_PATH = os.path.join(args.model_dir, "ddqn_cartpole_keras.weights.h5")
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    # Print configuration
    print(f"\n{'='*70}")
    print(f"Double DQN Training Configuration")
    print(f"{'='*70}")
    print(f"Environment:      {ENV_NAME}")
    print(f"Total Steps:      {args.total_steps:,}")
    print(f"Stage Size:       {args.stage_size:,}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Gamma:            {args.gamma}")
    print(f"Buffer Size:      {args.buffer_size:,}")
    print(f"Warmup Steps:     {args.warmup_steps:,}")
    print(f"Seed:             {args.seed}")
    print(f"Model Directory:  {args.model_dir}")
    print(f"Video Recording:  {'Disabled' if args.no_video else 'Enabled'}")
    print(f"TensorFlow:       {tf.__version__}")
    print(f"GPU Available:    {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"{'='*70}\n")
    
    # Train agent
    agent, reward_progress, std_progress = train_ddqn(
        env_name=ENV_NAME,
        total_steps=args.total_steps,
        stage_size=args.stage_size,
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        model_path=MODEL_PATH,
        video_dir=VIDEO_DIR,
        enable_video=not args.no_video,
        seed=args.seed,
        buffer_size=args.buffer_size,
        warmup_steps=args.warmup_steps
    )
    
    # Plot results
    stages = len(reward_progress)
    plot_training_progress(reward_progress, std_progress, stages, 
                          agent.episode_rewards, args.model_dir)
    
    # Final evaluation
    results, rewards = evaluate_and_report(agent, ENV_NAME, 
                                          n_eval_episodes=args.eval_episodes)
    
    # Save summary
    summary_path = os.path.join(args.model_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Double DQN Training Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Total Steps: {args.total_steps:,}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Gamma: {args.gamma}\n")
        f.write(f"  Buffer Size: {args.buffer_size:,}\n")
        f.write(f"  Warmup Steps: {args.warmup_steps:,}\n")
        f.write(f"  Seed: {args.seed}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Final Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
        f.write(f"  Total Episodes: {len(agent.episode_rewards)}\n")
        f.write(f"  Final Epsilon: {agent.epsilon:.4f}\n")
        f.write(f"  Solved: {'YES' if np.mean(rewards) >= 195.0 else 'NO'}\n\n")
        f.write(f"Evaluation Outcomes ({args.eval_episodes} episodes):\n")
        for k, v in results.items():
            percentage = (v / args.eval_episodes) * 100
            f.write(f"  {k}: {v} ({percentage:.1f}%)\n")
    
    print(f"📄 Saved training summary to {summary_path}")
    
    # Print final status
    final_reward = np.mean(rewards)
    if final_reward >= 195.0:
        print(f"\n🎉 SUCCESS! CartPole-v1 SOLVED with mean reward {final_reward:.2f}!\n")
        return 0
    else:
        print(f"\n⚠️  Training complete. Mean reward: {final_reward:.2f} (target: 195.0)\n")
        return 0  # Don't fail workflow, just report results


if __name__ == "__main__":
    sys.exit(main())
