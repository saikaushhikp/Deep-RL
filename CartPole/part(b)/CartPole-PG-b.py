import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Policy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
    
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)

def compute_returns(rewards, gamma, reward_to_go=False):
    """Compute discounted returns."""
    if reward_to_go:
        returns = []
        future_return = 0
        for r in reversed(rewards):
            future_return = r + gamma * future_return
            returns.insert(0, future_return)
        return returns
    else:
        total_return = sum([gamma**i * r for i, r in enumerate(rewards)])
        return [total_return] * len(rewards)

def plot_rewards(all_mean_rewards, fname):
    k = 50
    running_avg = np.convolve(all_mean_rewards, np.ones(k)/k, mode='valid')
    plt.figure(figsize=(8, 4))
    plt.plot(all_mean_rewards, label="All Returns", alpha=0.5)
    plt.plot(np.arange(k-1, len(all_mean_rewards)), running_avg, label=f"Running Average (k={k})", color='r', linewidth=2)
    plt.title(f"{fname}")
    plt.xlabel("Iteration")
    plt.ylabel("Average Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname + ".png")
    print(f"Plot saved as {fname}.png\n")
    plt.close()

def run_training(env_name="CartPole-v1", iterations=100, batch_size=2000,lr=1e-2, gamma=0.99, reward_to_go=True, advantage_norm=True,hidden_dim=128,):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PG training on {env_name} | reward_to_go={reward_to_go} advantage_norm={advantage_norm}  | device = {device}")
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    policy = Policy(state_dim, n_actions, hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_mean_rewards = []

    for it in range(iterations):
        batch_states, batch_actions, batch_weights, batch_episode_rewards = [], [], [], []
        steps = 0

        # collect trajectories until we reach desired batch size
        while steps < batch_size:
            obs, info = env.reset()
            done = False
            states, actions, rewards = [], [], []
            while True:
                s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(s_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())

                next_obs, reward, terminated, truncated, info = env.step(action)
                done_flag = terminated or truncated

                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                obs = next_obs

                if done_flag:
                    break

            steps += len(states)
            batch_states += states
            batch_actions += actions
            batch_episode_rewards.append(sum(rewards))
            batch_weights += list(compute_returns(rewards, gamma, reward_to_go))

        # prepare tensors
        batch_states_t = torch.tensor(np.array(batch_states), dtype=torch.float32, device=device)
        batch_actions_t = torch.tensor(batch_actions, dtype=torch.int64, device=device)
        batch_weights_t = torch.tensor(batch_weights, dtype=torch.float32, device=device)

        # advantage normalization
        if advantage_norm:
            mean = batch_weights_t.mean()
            std = batch_weights_t.std() + 1e-8
            batch_weights_t = (batch_weights_t - mean) / std

        logits = policy(batch_states_t)
        dists = torch.distributions.Categorical(logits=logits)
        logp = dists.log_prob(batch_actions_t)
        loss = -(logp * batch_weights_t).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_reward = np.mean(batch_episode_rewards)
        all_mean_rewards.append(mean_reward)
        print(f"Iteration {it+1}/{iterations} | Mean Reward: {mean_reward:.2f} | Episodes: {len(batch_episode_rewards)}")

    return all_mean_rewards


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

    all_mean_rewards = run_training(
        env_name=args.environment,
        iterations=args.iterations,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        reward_to_go=args.reward_to_go,
        advantage_norm=args.advantage_norm,
        hidden_dim=args.hidden_dim
    )

    fname = f"{args.environment}_PG_iters{args.iterations}_bs{args.batch_size}_g_lr_{'rtg' if args.reward_to_go else 'tot'}_{'advnorm' if args.advantage_norm else 'noadv'}_"
    plot_rewards(all_mean_rewards, fname)
