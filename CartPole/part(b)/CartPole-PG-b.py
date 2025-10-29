import argparse, os, time, random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        layers += [nn.Linear(prev_dim, act_dim), nn.Softmax(dim=-1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


def discount_cumsum(rewards, gamma):
    """Compute discounted cumulative rewards."""
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        discounted[t] = running_add
    return list(discounted)


def get_obs(obs):
    return np.array(obs[0] if isinstance(obs, (tuple, list)) else obs)


def step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, term, trunc, info = out
        done = term or trunc
    else:
        obs, reward, done, info = out
    return obs, reward, done, info


def run_training(env_name, iterations, batch_size, lr, gamma,
                 reward_to_go, advantage_norm, reward_scale,
                 reward_clip, hidden_dim, device, fname):



    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = Policy(obs_dim, act_dim, hidden_sizes=(hidden_dim, hidden_dim)).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    print(f"\nTraining {env_name} | γ={gamma} | lr={lr} | batch={batch_size} | device={device}")

    all_returns = []

    for it in range(iterations):
        batch_obs, batch_acts, batch_weights = [], [], []
        ep_returns = []
        log_probs = []

        steps_collected = 0
        while steps_collected < batch_size:
            obs = get_obs(env.reset())
            ep_rewards, ep_logps, ep_obs, ep_acts = [], [], [], []

            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                probs = policy(obs_tensor)
                dist = Categorical(probs)
                action = dist.sample()

                next_obs, reward, done, _ = step_env(env, action.item())
                if reward_scale != 1.0:
                    reward *= reward_scale
                if reward_clip:
                    reward = np.clip(reward, reward_clip[0], reward_clip[1])

                ep_obs.append(obs)
                ep_acts.append(action.item())
                ep_logps.append(dist.log_prob(action))
                ep_rewards.append(reward)
                obs = get_obs(next_obs)

            # Compute returns
            if reward_to_go:
                ep_returns = discount_cumsum(ep_rewards, gamma)
            else:
                G = sum([gamma**t * r for t, r in enumerate(ep_rewards)])
                ep_returns = [G for _ in ep_rewards]

            batch_obs += ep_obs
            batch_acts += ep_acts
            batch_weights += list(ep_returns)
            log_probs += ep_logps
            ep_returns_sum = sum(ep_rewards)
            ep_returns.append(ep_returns_sum)
            ep_returns = np.array(ep_returns)

            steps_collected += len(ep_rewards)

        # Convert to tensors
        log_probs_tensor = torch.stack(log_probs).to(device)
        advantages = torch.tensor(batch_weights, dtype=torch.float32, device=device)

        # Baseline + normalization
        if advantage_norm:
            advantages -= advantages.mean()
            advantages /= (advantages.std(unbiased=False) + 1e-8)

        # Policy gradient loss
        loss = (-(log_probs_tensor * advantages)).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        avg_return = np.mean(ep_returns)
        all_returns.append(avg_return)


        print(f"Iter {it+1:4d}/{iterations} | AvgReturn: {avg_return:8.2f} | Loss: {loss:.4f}")

    env.close()
    plt.figure(figsize=(8, 4))
    plt.plot(all_returns)
    plt.title(f"{fname}")
    plt.xlabel("Iteration")
    plt.ylabel("Average Return")
    # plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(fname)
    print(f"Plot saved as {fname}\n")
    return all_returns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="CartPole-v1")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward_to_go", action="store_true")
    parser.add_argument("--advantage_norm", action="store_true")
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--reward_clip_min", type=float, default=None)
    parser.add_argument("--reward_clip_max", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    clip_tuple = None
    if args.reward_clip_min is not None or args.reward_clip_max is not None:
        lo = -np.inf if args.reward_clip_min is None else args.reward_clip_min
        hi = np.inf if args.reward_clip_max is None else args.reward_clip_max
        clip_tuple = (lo, hi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fname = f"{args.environment}_PG_iters{args.iterations}_bs{args.batch_size}_g_lr_{'rtg' if args.reward_to_go else 'tot'}_{'advnorm' if args.advantage_norm else 'noadv'}_.png"
    run_training(
        env_name=args.environment,
        iterations=args.iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        reward_to_go=args.reward_to_go,
        advantage_norm=args.advantage_norm,
        reward_scale=args.reward_scale,
        reward_clip=clip_tuple,
        hidden_dim=args.hidden_dim,
        device=device,
        fname=fname
    )
