
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from itertools import count

def parse_args():
    p = argparse.ArgumentParser(description="Running with different episodic counts and mean rewards")
    p.add_argument("--environment", type=str, default="MountainCar-v0", help="Gym environment id")
    p.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    p.add_argument("--mean_n", type=int, default=5, help="n for rolling mean plot")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument("--render", action="store_true", help="Render environment (slows down execution)")
    return p.parse_args()


def safe_reset(env):
    """Handle gym vs gymnasium reset return types."""
    out = env.reset()
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out

def safe_step(env, action):
    """Handle gym vs gymnasium step return types."""
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    else:
        raise RuntimeError("Unexpected step output format: len = {}".format(len(out)))

def get_state_tensor(obs):
    """Return numpy array representation for plotting / visualization. Kept simple."""
    return np.array(obs, dtype=np.float32)

def select_random_action(action_space):
    """Return a single integer action chosen uniformly at random."""
    return action_space.sample()

def plot_results(rewards_mean, steps, best_rewards_mean, env_name, file_name, action_scatter):
    """Create a 2-panel plot: performance and action choices scatter."""
    fig = plt.figure(figsize=(12,5))

    ax1 = fig.add_subplot(121)
    ax1.plot(steps, rewards_mean, label=f"{len(steps)}-point rolling mean")
    ax1.plot(steps, best_rewards_mean, label="Best mean reward")
    ax1.grid(True)
    ax1.set_xlabel("Total environment steps")
    ax1.set_ylabel("Reward (higher is better)")
    ax1.legend()
    ax1.set_title(f"Performance of random agent on {env_name}")

    ax2 = fig.add_subplot(122)
    if len(action_scatter) > 0:
        arr = np.array(action_scatter)
        X = arr[:,0].astype(float)
        Y = arr[:,1].astype(float)
        Z = arr[:,2].astype(int)
        # color map for 3 discrete actions
        cmap = {0: 'lime', 1: 'red', 2: 'blue'}
        colors = [cmap[int(a)] for a in Z]
        ax2.scatter(X, Y, c=colors, s=12, alpha=0.7)
        action_names = ['Left (0)', 'No-Op (1)', 'Right (2)']
        # legend patches
        legend_recs = [mpatches.Rectangle((0,0),1,1,fc=cmap[i]) for i in range(3)]
        ax2.legend(legend_recs, action_names, loc='best')
    ax2.set_title("Random agent action choices (sampled states)")
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Velocity")

    plt.suptitle(f"{env_name} - Random Agent Analysis")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(file_name, dpi=200)
    print(f"Saved plot to {file_name}")
    plt.show()
    plt.close(fig)
    return

def run_random_agent(env_id, episodes, mean_n, seed=None, render=False):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    env = gym.make(env_id)
    print("Environment:", env_id)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    episode_rewards = []
    best_reward = -float('inf')
    rewards_mean = []
    best_rewards_mean = []
    steps = []

    total_steps = 0
    success_count = 0

    action_scatter = []  # store (pos, vel, action) samples for plotting

    for ep in range(episodes):
        obs = safe_reset(env)
        state = get_state_tensor(obs)
        total_reward = 0.0

        for t in count():
            if render:
                env.render()

            action = select_random_action(env.action_space)
            if len(action_scatter) < 2000 and (total_steps % max(1, int(100/episodes)) == 0):
                action_scatter.append((state[0], state[1], action))

            obs, reward, done, info = safe_step(env, action)
            state = get_state_tensor(obs)
            total_reward += reward
            total_steps += 1

            if done or t >= 10000:  # safety cap
                # success condition: position >= 0.5 at termination
                try:
                    pos = state[0]
                except:
                    pos = None
                if pos is not None and pos >= 0.5:
                    success_count += 1
                break

        episode_rewards.append(total_reward)

        if len(episode_rewards) >= mean_n:
            present_mean = float(np.mean(episode_rewards[-mean_n:]))
            rewards_mean.append(present_mean)
            best_reward = max(present_mean, best_reward)
            best_rewards_mean.append(best_reward)
            steps.append(total_steps)

        print(f"Episode {ep+1}/{episodes} | Reward = {total_reward:.2f} | Successes so far = {success_count}")

    env.close()

    fn = f"{env_id}_random_{episodes}ep_mean{mean_n}.png"
    plot_results(rewards_mean, steps, best_rewards_mean, env_id, fn, action_scatter)

    # Summary
    summary = {
        "total_episodes": episodes,
        "mean_n": mean_n,
        "final_mean_reward": rewards_mean[-1] if rewards_mean else None,
        "best_mean_reward": best_rewards_mean[-1] if best_rewards_mean else None,
        "success_count": success_count,
        "total_steps": total_steps
    }
    return summary

if __name__ == "__main__":
    args = parse_args()
    summary = run_random_agent(args.environment, args.episodes, args.mean_n, args.seed, args.render)
    print()
    print("="*10 +"Summary" + "="*10)
    for k, v in summary.items():
        print(f"{k}: {v}")