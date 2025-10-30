"""
Hybrid Policy Gradient Implementation
- Combines fast step-based batching (Implim-1)
- with stable, smooth learning (Implim-2 style advantage computation)
"""

import argparse
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import gymnasium as gym


# ----------------------------
# Policy Network
# ----------------------------
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Helpers
# ----------------------------
def discount_cumsum(rewards, gamma):
    """Compute discounted cumulative sums of rewards."""
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        discounted[t] = running_add
    return discounted


def get_obs(obs):
    """Handle gymnasium reset() returning (obs, info)"""
    if isinstance(obs, (tuple, list)):
        return np.array(obs[0])
    return np.array(obs)


def step_env(env, action):
    """Handle gymnasium step() possibly returning 4 or 5 outputs."""
    out = env.step(action)
    if len(out) == 5:
        obs, reward, term, trunc, info = out
        done = term or trunc
    else:
        obs, reward, done, info = out
    return obs, reward, done, info


def plot_rewards(rewards, fname):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label="Average Return per Iteration")
    plt.title(os.path.basename(fname).replace(".png", ""))
    plt.xlabel("Iteration")
    plt.ylabel("Average Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Plot saved as {fname}\n")


# ----------------------------
# Hybrid Training Function
# ----------------------------
def run_training(env_name, iterations, batch_size, lr, gamma,
                 reward_to_go, advantage_norm, reward_scale,
                 reward_clip, hidden_dim, seed, device):

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = Policy(obs_dim, act_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    print(f"\nTraining {env_name} | γ={gamma} | lr={lr} | batch={batch_size} | device={device}")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")

    all_avg_returns = []

    for it in range(iterations):
        episodes = []   # [(logps, returns, total_reward), ...]
        steps_collected = 0

        # Collect step-based batch (like Implim-1)
        while steps_collected < batch_size:
            obs = get_obs(env.reset())
            ep_rewards, ep_logps = [], []
            done = False

            while not done:
                # obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                # with torch.no_grad():
                #     probs = policy(obs_tensor)
                # dist = Categorical(probs)
                # action = dist.sample()
                # logp = dist.log_prob(action)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                probs = policy(obs_tensor)  # keep graph for log_prob gradient
                dist = Categorical(probs)
                action = dist.sample()
                logp = dist.log_prob(action)


                next_obs, reward, done, _ = step_env(env, action.item())

                # reward scaling/clipping
                if reward_scale != 1.0:
                    reward *= reward_scale
                if reward_clip is not None:
                    reward = np.clip(reward, reward_clip[0], reward_clip[1])

                ep_rewards.append(float(reward))
                ep_logps.append(logp)
                obs = get_obs(next_obs)

            # Compute returns for this episode
            if reward_to_go:
                ep_returns = discount_cumsum(ep_rewards, gamma)
            else:
                total_R = sum([gamma ** t * r for t, r in enumerate(ep_rewards)])
                ep_returns = np.full_like(ep_rewards, fill_value=total_R, dtype=np.float32)

            episodes.append((ep_logps, ep_returns, sum(ep_rewards)))
            steps_collected += len(ep_rewards)

        # Flatten all episodes for optimization (like Implim-2)
        flat_logps = torch.cat([torch.stack(ep[0]) for ep in episodes])
        flat_returns = torch.tensor(np.concatenate([ep[1] for ep in episodes]),
                                    dtype=torch.float32, device=device)

        # Normalize advantages if needed
        if advantage_norm:
            flat_returns = (flat_returns - flat_returns.mean()) / (flat_returns.std(unbiased=False) + 1e-8)

        # Compute loss
        loss = -(flat_logps * flat_returns).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Logging
        avg_return = np.mean([ep[2] for ep in episodes])
        all_avg_returns.append(avg_return)

        if (it + 1) % max(1, iterations // 10) == 0 or it < 5:
            print(f"Iter {it+1:4d}/{iterations} | AvgReturn: {avg_return:8.2f} | Loss: {loss.item():.4f}")

    env.close()

    # Plot learning curve
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fname = f"{env_name}_PG_HYBRID_{timestr}.png"
    plot_rewards(all_avg_returns, fname)

    return fname, all_avg_returns


# ----------------------------
# Command-line Interface
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="CartPole-v1")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_to_go", action="store_true")
    parser.add_argument("--advantage_norm", action="store_true")
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--reward_clip_min", type=float, default=None)
    parser.add_argument("--reward_clip_max", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # reward clipping
    reward_clip = None
    if args.reward_clip_min is not None or args.reward_clip_max is not None:
        lo = -np.inf if args.reward_clip_min is None else args.reward_clip_min
        hi = np.inf if args.reward_clip_max is None else args.reward_clip_max
        reward_clip = (lo, hi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_training(
        env_name=args.environment,
        iterations=args.iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        reward_to_go=args.reward_to_go,
        advantage_norm=args.advantage_norm,
        reward_scale=args.reward_scale,
        reward_clip=reward_clip,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=device
    )
