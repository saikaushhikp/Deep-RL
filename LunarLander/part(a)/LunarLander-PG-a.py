
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import random
import argparse
from itertools import count



parser = argparse.ArgumentParser(description="Policy Gradient for LunarLander-v3")
parser.add_argument("--environment", type=str, required=True,
                    help="Gym environment name (e.g., LunarLander-v3")
parser.add_argument("--iterations", type=int, default=50,
                    help="Number of iterations (each iteration = batch of episodes)")
parser.add_argument("--batch", type=int, default=32,
                    help="Batch size (number of episodes per iteration)")
args = parser.parse_args()



def plot_rewards(rewards, filename):
    """Plot the average total rewards per iteration."""
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(rewards, label='Average Reward per Iteration')
    plt.title(f"Random Agent Performance on {args.environment}")
    plt.xlabel("Iteration")
    plt.ylabel("Average Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def get_state(obs):
    """Convert observation (state) to PyTorch tensor."""
    state = torch.from_numpy(np.array(obs)).float().unsqueeze(0)
    return state


def select_random_action(env):
    """Select a random valid action from the environment."""
    return env.action_space.sample()


def run_random_agent(env, iterations, batch_size):
    """Run random agent for given environment and collect average rewards."""
    all_rewards = []

    for it in range(iterations):
        batch_rewards = []

        for ep in range(batch_size):
            obs = env.reset()
            # Compatible with Gym v0.26+
            if isinstance(obs, tuple):
                obs = obs[0]

            total_reward = 0
            for t in count():
                action = select_random_action(env)
                step_output = env.step(action)

                if len(step_output) == 5:
                    obs, reward, terminated, truncated, _ = step_output
                    done = terminated or truncated
                else:
                    obs, reward, done, _ = step_output

                total_reward += reward

                if done:
                    break

            batch_rewards.append(total_reward)

        avg_reward = np.mean(batch_rewards)
        print(f"Iteration {it+1}/{iterations}: Average Reward = {avg_reward:.2f}")
        all_rewards.append(avg_reward)

    env.close()
    return all_rewards


if __name__ == "__main__":
    # Load Gym environment
    env = gym.make(args.environment)
    print(f"\nLoaded Environment: {args.environment}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")

    # Run random agent
    rewards = run_random_agent(env, args.iterations, args.batch)

    # Save plot
    file_name = f"{args.environment}_{args.iterations}_{args.batch}_random_agent.png"
    plot_rewards(rewards, file_name)
    print(f"Plot saved as {file_name}")
