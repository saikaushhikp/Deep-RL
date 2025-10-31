import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py
from collections import deque
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Hyperparameters (from DQN paper)
LEARNING_RATE = 0.00025
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000  # Steps over which to decay epsilon
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 10000  # Update target network every N steps
FRAME_STACK = 4  # Stack 4 frames as per paper
LEARNING_STARTS = 50000  # Start learning after N steps
SAVE_FREQ = 100000  # Save model every N steps

# Frame preprocessing parameters
FRAME_WIDTH = 84
FRAME_HEIGHT = 84


class FramePreprocessor:
    """Preprocess frames as described in DQN paper"""
    
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height
        
    def process(self, frame):
        """
        Convert RGB to grayscale and resize to 84x84
        Args:
            frame: RGB frame from environment (210x160x3 for Atari)
        Returns:
            Processed grayscale frame (84x84)
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
            
        # Resize to 84x84
        resized = cv2.resize(gray, (self.width, self.height), 
                           interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized


class FrameStack:
    """Stack frames to capture temporal information"""
    
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        
    def reset(self, frame):
        """Reset with initial frame"""
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self.get_state()
    
    def update(self, frame):
        """Add new frame and return stacked state"""
        self.frames.append(frame)
        return self.get_state()
    
    def get_state(self):
        """Return stacked frames as numpy array"""
        return np.stack(self.frames, axis=0)


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch from buffer"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNetwork(nn.Module):
    """Deep Q-Network with CNN architecture from DQN paper"""
    
    def __init__(self, num_actions, frame_stack=4):
        super(DQNetwork, self).__init__()
        
        # Convolutional layers as per DQN paper
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        # Input: 84x84, after conv1: 20x20, after conv2: 9x9, after conv3: 7x7
        conv_output_size = 64 * 7 * 7
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def forward(self, x):
        """Forward pass through network"""
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class DQNAgent:
    """DQN Agent with experience replay and target network"""
    
    def __init__(self, num_actions, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_actions = num_actions
        
        # Q-network and target network
        self.q_network = DQNetwork(num_actions, FRAME_STACK).to(device)
        self.target_network = DQNetwork(num_actions, FRAME_STACK).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        
        # Training stats
        self.steps = 0
        self.epsilon = EPSILON_START
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()
    
    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * self.steps / EPSILON_DECAY
        )
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * GAMMA * max_next_q_values
        
        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping as per paper
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_dqn(num_episodes=2000, max_steps_per_episode=10000):
    """Train DQN agent on Pong-v5"""
    
    # Initialize environment
    env = gym.make('ALE/Pong-v5', render_mode=None)
    
    # Get action space size
    num_actions = env.action_space.n
    
    # Initialize components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    agent = DQNAgent(num_actions, device)
    preprocessor = FramePreprocessor(FRAME_WIDTH, FRAME_HEIGHT)
    frame_stack = FrameStack(FRAME_STACK)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    mean_rewards = []
    best_mean_reward = -float('inf')
    
    global_step = 0
    
    print("Starting training...")
    print(f"Episodes: {num_episodes}")
    print(f"Learning starts at step: {LEARNING_STARTS}")
    
    for episode in range(num_episodes):
        # Reset environment
        obs, info = env.reset()
        processed_frame = preprocessor.process(obs)
        state = frame_stack.reset(processed_frame)
        
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Preprocess next frame
            processed_next_frame = preprocessor.process(next_obs)
            next_state = frame_stack.update(processed_next_frame)
            
            # Clip reward as per DQN paper
            clipped_reward = np.clip(reward, -1, 1)
            
            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, clipped_reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            global_step += 1
            agent.steps = global_step
            
            # Update epsilon
            agent.update_epsilon()
            
            # Train agent
            if global_step >= LEARNING_STARTS:
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)
                
                # Update target network
                if global_step % TARGET_UPDATE_FREQ == 0:
                    agent.update_target_network()
                    print(f"Updated target network at step {global_step}")
            
            if done:
                break
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Calculate moving average (last 100 episodes)
        if len(episode_rewards) >= 100:
            mean_reward = np.mean(episode_rewards[-100:])
            mean_rewards.append(mean_reward)
            
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            avg_loss = np.mean(losses[-10:]) if losses else 0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Steps: {global_step} | "
                  f"Avg Reward (10 ep): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Avg Loss (10 ep): {avg_loss:.4f}")
            
            if len(episode_rewards) >= 100:
                print(f"  Mean Reward (100 ep): {mean_rewards[-1]:.2f} | "
                      f"Best Mean: {best_mean_reward:.2f}")
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'mean_rewards': mean_rewards,
        'best_mean_reward': best_mean_reward,
        'agent': agent
    }


def plot_learning_curves(results, n_episodes=100):
    """Plot learning curves"""
    
    episode_rewards = results['episode_rewards']
    mean_rewards = results['mean_rewards']
    best_mean_reward = results['best_mean_reward']
    
    # Calculate cumulative steps for x-axis
    # Approximate steps per episode (varies, but we'll use a running sum)
    steps_per_episode = []
    cumulative_steps = 0
    
    # Estimate steps (rough approximation for plotting)
    for i in range(len(episode_rewards)):
        # Early episodes might be longer, later ones shorter as agent improves
        estimated_steps = 1000  # Average estimate
        cumulative_steps += estimated_steps
        steps_per_episode.append(cumulative_steps)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Episode rewards and mean rewards
    ax1 = axes[0]
    
    # Plot individual episode rewards (lighter)
    ax1.plot(steps_per_episode, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Plot mean n-episode rewards
    if mean_rewards:
        mean_steps = steps_per_episode[n_episodes-1:]
        ax1.plot(mean_steps, mean_rewards, color='red', linewidth=2, 
                label=f'Mean {n_episodes}-Episode Reward')
        
        # Plot best mean reward line
        ax1.axhline(y=best_mean_reward, color='green', linestyle='--', 
                   linewidth=2, label=f'Best Mean Reward: {best_mean_reward:.2f}')
    
    ax1.set_xlabel('Time Steps (×1000)', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('DQN Training Performance on Pong-v5', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis to show in thousands
    ax1.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    
    # Plot 2: Training loss
    ax2 = axes[1]
    if results['losses']:
        loss_steps = np.linspace(0, steps_per_episode[-1], len(results['losses']))
        ax2.plot(loss_steps, results['losses'], alpha=0.6, color='orange')
        ax2.set_xlabel('Time Steps (×1000)', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('dqn_pong_learning_curves.png', dpi=300, bbox_inches='tight')
    print("\nLearning curves saved as 'dqn_pong_learning_curves.png'")
    # plt.show()


def evaluate_agent(agent, num_episodes=10):
    """Evaluate trained agent"""
    
    env = gym.make('ALE/Pong-v5', render_mode=None)
    preprocessor = FramePreprocessor(FRAME_WIDTH, FRAME_HEIGHT)
    frame_stack = FrameStack(FRAME_STACK)
    
    eval_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        processed_frame = preprocessor.process(obs)
        state = frame_stack.reset(processed_frame)
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            processed_next_frame = preprocessor.process(next_obs)
            state = frame_stack.update(processed_next_frame)
            
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}: Reward = {episode_reward}")
    
    env.close()
    
    mean_eval_reward = np.mean(eval_rewards)
    print(f"\nMean Evaluation Reward: {mean_eval_reward:.2f}")
    
    return eval_rewards


if __name__ == "__main__":
    print("=" * 60)
    print("DQN Training on Pong-v5")
    print("=" * 60)
    print("\nNote: This is a simplified training run.")
    print("For full convergence, you may need 5M-10M steps (many hours).")
    print("Adjust num_episodes for longer training.\n")
    
    # Train agent (reduce episodes for quick testing)
    # For full training, use num_episodes=2000 or more
    results = train_dqn(num_episodes=500, max_steps_per_episode=10000)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Mean Reward: {results['best_mean_reward']:.2f}")
    
    # Plot learning curves
    plot_learning_curves(results, n_episodes=100)
    
    # Evaluate agent
    print("\n" + "=" * 60)
    print("Evaluating Trained Agent")
    print("=" * 60)
    eval_rewards = evaluate_agent(results['agent'], num_episodes=10)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)