import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("ALE/Pong-v5", render_mode=None)  # set render_mode='human' if you want to see the game

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Number of possible actions:", env.action_space.n)

num_episodes = 15
episode_rewards = []
reward_trajectories = []  # store reward progression per episode

for ep in range(num_episodes):
    obs, info = env.reset(seed=ep)
    done = False
    total_reward = 0
    rewards = []

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(total_reward)  # cumulative reward curve
        done = terminated or truncated

    episode_rewards.append(total_reward)
    reward_trajectories.append(rewards)
    print(f"Episode {ep+1}: Total Reward = {total_reward}")

env.close()


print(f"\nAverage total reward over {num_episodes} random episodes:", np.mean(episode_rewards))

plt.figure(figsize=(8, 5))
for i, rewards in enumerate(reward_trajectories):
    plt.plot(rewards, label=f'Episode {i+1}')
plt.title("Cumulative Reward per Step (Random Agent on Pong-v5)")
plt.xlabel("Time Steps")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(f"pong_{num_episodes}_random_agent_rewards.png")
# plt.show()
