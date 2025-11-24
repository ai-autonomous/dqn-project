"""
Double DQN implementation for CartPole-v1 using Keras/TensorFlow.
Google Colab Compatible - Cell-by-Cell Execution

Includes:
  • Replay buffer with experience replay
  • Target network with periodic updates
  • Epsilon-greedy exploration with decay
  • Best-episode video recording
  • Reward progression & evaluation plots
  • Comprehensive termination analysis
"""

# ===================== REPLAY BUFFER =====================
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

# ===================== DDQN AGENT =====================
class DDQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay_steps=10_000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = (epsilon_start - epsilon_final) / epsilon_decay_steps
        
        # Networks
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.replay_buffer = ReplayBuffer(50_000)
        
        # Training stats
        self.training_step = 0
        self.episode_rewards = []
        
    def _build_network(self):
        """Build Q-network with 2 hidden layers of 128 units each"""
        model = keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_dim, activation='linear')
        ])
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
    
    def train_step(self, batch_size=32):
        """Perform one training step using Double DQN"""
        if self.replay_buffer.size() < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        with tf.GradientTape() as tape:
            # Current Q-values
            q_values = self.q_network(states, training=True)
            q_values_actions = tf.reduce_sum(
                q_values * tf.one_hot(actions, self.action_dim), axis=1
            )
            
            # Double DQN: use Q-network to select actions, target network to evaluate
            next_q_values = self.q_network(next_states, training=False)
            best_actions = tf.argmax(next_q_values, axis=1)
            
            target_q_values = self.target_network(next_states, training=False)
            target_q_values_actions = tf.reduce_sum(
                target_q_values * tf.one_hot(best_actions, self.action_dim), axis=1
            )
            
            # Compute targets
            targets = rewards + (1 - dones) * self.gamma * target_q_values_actions
            
            # Loss
            loss = tf.reduce_mean(tf.square(targets - q_values_actions))
        
        # Update Q-network
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Decay epsilon
        if self.epsilon > self.epsilon_final:
            self.epsilon -= self.epsilon_decay
        
        self.training_step += 1
        return loss.numpy()
    
    def save(self, path):
        """Save model weights"""
        self.q_network.save_weights(path)
        print(f"💾 Model saved to {path}")
    
    def load(self, path):
        """Load model weights"""
        if os.path.exists(path):
            self.q_network.load_weights(path)
            self.update_target_network()
            print(f"📦 Model loaded from {path}")
            return True
        return False

# ===================== TERMINATION REASON =====================
def termination_reason(terminated, truncated, steps, max_steps):
    if truncated and steps == max_steps:
        return "SOLVED"
    elif truncated:
        return "TIME_LIMIT"
    elif terminated and steps >= 195:
        return "GOOD_RUN"
    elif terminated:
        return "FAIL"
    else:
        return "UNKNOWN"

# ===================== VIDEO RECORDING =====================
best_reward = -999999

def record_best_video(agent, env_name=ENV_NAME):
    """Record video of best episode only"""
    global best_reward
    
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, VIDEO_DIR, name_prefix="best_cartpole_episode")
    
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
    else:
        # Remove non-best video
        for f in os.listdir(VIDEO_DIR):
            if "best_cartpole_episode" in f:
                try:
                    os.remove(os.path.join(VIDEO_DIR, f))
                except:
                    pass

# ===================== TRAINING =====================
def train_ddqn(, stage_size, lr, batch_size, gamma):
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DDQNAgent(state_dim, action_dim, lr=lr, gamma=gamma, 
                      epsilon_decay_steps=int( * 0.1))
    
    # Load existing model if available
    agent.load(MODEL_PATH)
    
    stages = max(1,  // stage_size)
    reward_progress = []
    
    obs, _ = env.reset(seed=0)
    episode_reward = 0
    episode_steps = 0
    global_step = 0
    update_target_every = 250
    train_every = 4
    
    losses = []
    
    for stage in range(stages):
        print(f"\n=== 🚀 Stage {stage+1}/{stages} → training {stage_size:,} steps ===")
        stage_rewards = []
        
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
            if global_step % train_every == 0:
                loss = agent.train_step(batch_size)
                if loss > 0:
                    losses.append(loss)
            
            # Update target network
            if global_step % update_target_every == 0:
                agent.update_target_network()
            
            # Episode end
            if done:
                agent.episode_rewards.append(episode_reward)
                stage_rewards.append(episode_reward)
                obs, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
            else:
                obs = next_obs
        
        # Stage evaluation
        mean_reward = evaluate_agent(agent, n_episodes=10)
        reward_progress.append(mean_reward)
        
        print(f"📈 Eval: mean={mean_reward:.2f}, epsilon={agent.epsilon:.3f}, buffer={agent.replay_buffer.size()}")
        if losses:
            print(f"📊 Avg loss: {np.mean(losses[-1000:]):.4f}")
        
        # Save model
        agent.save(MODEL_PATH)
        
        # Record best video
        record_best_video(agent)
    
    env.close()
    
    # Plot training progress
    plot_training_progress(reward_progress, stages, agent.episode_rewards)
    
    return agent

# ===================== EVALUATION =====================
def evaluate_agent(agent, n_episodes=10, render=False):
    """Evaluate agent performance"""
    env = gym.make(ENV_NAME, render_mode="human" if render else None)
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

def evaluate_and_report(agent, n_eval_episodes=20):
    """Comprehensive evaluation with termination analysis"""
    env = gym.make(ENV_NAME)
    max_steps = env.spec.max_episode_steps
    
    results = {"SOLVED": 0, "GOOD_RUN": 0, "FAIL": 0, "TIME_LIMIT": 0, "UNKNOWN": 0}
    rewards = []
    
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
    
    print("\n=== 🧪 Evaluation Summary ===")
    for k, v in results.items():
        print(f"{k:>12}: {v}")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

# ===================== PLOTTING =====================
def plot_training_progress(stage_rewards, stages, episode_rewards):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Stage rewards
    ax1.plot(np.arange(1, stages + 1), stage_rewards, marker="o", linewidth=2)
    ax1.set_title("Double DQN Training Progress on CartPole-v1")
    ax1.set_xlabel("Training Stage")
    ax1.set_ylabel("Mean Evaluation Reward")
    ax1.grid(True, alpha=0.3)
    
    # Episode rewards (smoothed)
    if len(episode_rewards) > 0:
        window = min(100, len(episode_rewards) // 10)
        if window > 1:
            smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax2.plot(smoothed, linewidth=1.5, label='Smoothed')
        ax2.plot(episode_rewards, alpha=0.3, linewidth=0.5, label='Raw')
        ax2.set_title("Episode Rewards During Training")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episode Reward")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_reward_plot_ddqn_cartpole.png"), dpi=150)
    print(f"📊 Saved training plot to {MODEL_DIR}/training_reward_plot_ddqn_cartpole.png")

# ===================== MAIN =====================
if __name__ == "__main__":
    # Train agent
    agent = train_ddqn(
        =args.total_steps,
        stage_size=args.stage_size,
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma
    )
    
    # Final evaluation
    evaluate_and_report(agent, n_eval_episodes=20)
    
    print("🎉 Finished!")
