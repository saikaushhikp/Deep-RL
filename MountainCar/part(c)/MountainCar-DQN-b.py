import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import argparse

class QNetwork(nn.Module):
    """Deep Q-Network: 3-layer MLP for state-action value prediction."""
    def __init__(self, state_dimension, action_dimension):
        super().__init__()
        self.network_layers = nn.Sequential(
            nn.Linear(state_dimension, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dimension)
        )

    def forward(self, state):
        return self.network_layers(state)


def calculate_shaped_reward(state, environment_reward):
    """Enhances the reward signal by adding a velocity-based bonus."""
    car_velocity = state[1]
    velocity_bonus = abs(car_velocity) * 10  # Encourages maintaining speed
    return environment_reward + velocity_bonus


def train_agent(config):
    warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.functional')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Environment setup
    env = gym.make('MountainCar-v0')
    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.n

    # Initialize networks
    online_network = QNetwork(state_dimension, action_dimension).to(device)
    target_network = QNetwork(state_dimension, action_dimension).to(device)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(online_network.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    experience_buffer = deque(maxlen=config.memory_size)

    # Training metrics
    episode_rewards = []
    exploration_rate = config.epsilon_start
    total_timesteps = 0


    for episode in range(config.num_episodes):
        if episode % 10 == 0:
            print(f"Episode {episode}/{config.num_episodes}")

        state_np, _ = env.reset()
        state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        episode_reward = 0
        episode_steps = 0

        while True:
            if random.random() < exploration_rate:
                action = env.action_space.sample()  # Random exploration
            else:
                with torch.no_grad():
                    action = online_network(state_tensor).argmax().item()  # Greedy exploitation

            next_state_np, env_reward, terminated, truncated, _ = env.step(action)
            shaped_reward = calculate_shaped_reward(next_state_np, env_reward)
            next_state_tensor = torch.FloatTensor(next_state_np).unsqueeze(0).to(device)
            episode_done = terminated or truncated

            experience_buffer.append((state_tensor, action, shaped_reward, next_state_tensor, episode_done))

            state_tensor = next_state_tensor
            episode_reward += shaped_reward
            episode_steps += 1
            total_timesteps += 1

            if len(experience_buffer) > config.batch_size:
                batch = random.sample(experience_buffer, config.batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)

                states_tensor = torch.cat(batch_states).to(device)
                next_states_tensor = torch.cat(batch_next_states).to(device)
                actions_tensor = torch.tensor(batch_actions).unsqueeze(1).to(device)
                rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                dones_tensor = torch.tensor(batch_dones, dtype=torch.bool).to(device)

                # Compute current Q-values
                current_q_values = online_network(states_tensor).gather(1, actions_tensor)

                # Compute target Q-values using target network
                with torch.no_grad():
                    next_q_values = target_network(next_states_tensor).max(1)[0]
                    target_q_values = rewards_tensor + config.gamma * next_q_values * (~dones_tensor).float()

                # Compute loss and update online network
                loss = criterion(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if episode_done:
                break

        if exploration_rate > config.epsilon_min:
            exploration_rate *= config.epsilon_decay_factor

        if (episode + 1) % config.target_update_episodes == 0:
            target_network.load_state_dict(online_network.state_dict())

        episode_rewards.append(episode_reward)

        if len(episode_rewards) >= config.n_mean_episodes and episode % 10 == 0:
            recent_mean_reward = np.mean(episode_rewards[-config.n_mean_episodes:])
            print(f"Mean {config.n_mean_episodes}-episode reward: {recent_mean_reward:.2f}")

    env.close()
    return episode_rewards, online_network, device




def visualize_training_results(config, training_rewards, trained_network, device):
    # Prepare training metrics
    episode_count = config.num_episodes
    window_size = config.n_mean_episodes
    batch_size = config.batch_size

    episodes = list(range(1, len(training_rewards) + 1))
    smoothed_rewards = []
    peak_rewards = []

    highest_mean = -float('inf')
    for i in range(len(training_rewards)):
        if i >= window_size:
            window_mean = np.mean(training_rewards[i - window_size:i])
            smoothed_rewards.append(window_mean)
            highest_mean = max(highest_mean, window_mean)
            peak_rewards.append(highest_mean)
        else:
            smoothed_rewards.append(None)
            peak_rewards.append(None)

    output_filename = f"{config.environment}_DQN_{episode_count}_episodes_{batch_size}batch.png"

    _, (performance_plot, policy_plot) = plt.subplots(1, 2, figsize=(12, 5))

    performance_plot.plot(episodes, smoothed_rewards, label=f"{window_size}-episode mean", color="blue", alpha = 0.7)
    performance_plot.plot(episodes, peak_rewards, label="Best mean reward", color="orange", linestyle="--")
    performance_plot.set_xlabel("Episode")
    performance_plot.set_ylabel("Reward")
    performance_plot.set_title(f"Learning Progress on {config.environment}")
    performance_plot.grid(True)
    performance_plot.legend()

    position_range = np.linspace(-1.5, 0.6, 50)  # Car position range
    velocity_range = np.linspace(-1, 1, 50)      # Car velocity range
    positions, velocities = np.meshgrid(position_range, velocity_range)
    states = np.array(list(zip(positions.flatten(), velocities.flatten())))

    state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        action_choices = trained_network(state_tensor).argmax(dim=1).cpu().numpy()

    action_colors = ['lime', 'red', 'blue']  # left, no-op, right
    policy_plot.scatter(states[:, 0], states[:, 1], 
                       c=[action_colors[a] for a in action_choices], 
                       s=1, alpha=0.7)
    policy_plot.set_xlabel("Car Position")
    policy_plot.set_ylabel("Car Velocity")
    policy_plot.set_title("Learned Policy Map")

    action_labels = ["Left", "No Action", "Right"]
    legend_patches = [mpatches.Patch(color=action_colors[i], 
                                   label=action_labels[i]) 
                     for i in range(3)]
    policy_plot.legend(handles=legend_patches)
    policy_plot.set_xlim([-1.5, 0.6])
    policy_plot.set_ylim([-1, 1])

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()
    print(f"Visualization saved as {output_filename}")

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Deep Q-Learning for Mountain Car Environment")

    # Training configuration parameters
    parser.add_argument("--environment", type=str, default="MountainCar-v0",help="Gymnasium environment to train on")
    parser.add_argument("--learning_rate", type=float, default=5e-4,help="Learning rate for the neural network")
    parser.add_argument("--gamma", type=float, default=0.99,help="Discount factor for future rewards")
    parser.add_argument("--epsilon_start", type=float, default=1.0,help="Initial exploration rate")
    parser.add_argument("--epsilon_min", type=float, default=0.01,help="Minimum exploration rate")
    parser.add_argument("--epsilon_decay_factor", type=float, default=0.997,help="Rate at which exploration decreases")
    parser.add_argument("--memory_size", type=int, default=10000,help="Size of experience replay buffer")
    parser.add_argument("--batch_size", type=int, default=64,help="Number of experiences to learn from at once")
    parser.add_argument("--target_update_episodes", type=int, default=20,help="Episodes between target network updates")
    parser.add_argument("--num_episodes", type=int, default=1000,help="Total number of training episodes")
    parser.add_argument("--n_mean_episodes", type=int, default=50,help="Window size for calculating mean reward")

    # Parse arguments and start training
    config = parser.parse_args()

    # Train the agent
    training_history, trained_network, device = train_agent(config)
    
    # Visualize and save results
    visualize_training_results(config, training_history, trained_network, device)
