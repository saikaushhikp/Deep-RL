
import math
import random
from itertools import count
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import pandas as pd

#################################
# Argument Parsing
#################################
parser = argparse.ArgumentParser(description="DQN for MountainCar-v0")
parser.add_argument("--environment", type=str, default="MountainCar-v0")
parser.add_argument("--num_episodes", type=int, default=200)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--mean_n", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

###############
# Setup
###############
env = gym.make(args.environment)
env.reset(seed=args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################
# DQN Model Definition
##############3#######
class DQN(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=200, action_dim=3):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

####################
# Replay Memory
####################
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#####################
# Helper Functions
#####################
def get_state(obs):
    """Convert observation to tensor."""
    s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    return s

def select_action(state, eps_threshold):
    """ε-greedy action selection."""
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def optimize_model():
    """Perform a single optimization step."""
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state).to(dtype=torch.float32, device=device)
    action_batch = torch.cat(batch.action).to(dtype=torch.long, device=device)
    reward_batch = torch.cat(batch.reward).to(dtype=torch.float32, device=device)

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool
    )

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    ).to(dtype=torch.float32, device=device)

    # Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s_{t+1})
    next_state_values = torch.zeros(batch_size, device=device, dtype=torch.float32)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # loss
    loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

##################
# Training Loop
###################
n_actions = env.action_space.n
batch_size = args.batch
gamma = args.gamma
num_episodes = args.num_episodes
lr = args.learning_rate
mean_n = args.mean_n

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = ReplayMemory(10000)

eps_start, eps_end, eps_decay = 1.0, 0.02, 800
target_update = 10
initial_memory = 1000
total_steps = 0

rewards_list, mean_rewards, best_mean_rewards, steps_list = [], [], [], []
best_mean = -float("inf")
successes = 0

for i_episode in range(num_episodes):
    obs, _ = env.reset()
    state = get_state(obs)
    total_reward = 0.0

    for t in count():
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * total_steps / eps_decay)
        action = select_action(state, eps_threshold)
        obs_next, reward, done, truncated, _ = env.step(action.item())
        done = done or truncated

        # --- Reward shaping: ---
        # Original reward is always -1 until goal is reached
        # We shape it to encourage progress towards the goal
        shaped_reward = obs_next[0] + 0.5 + abs(obs_next[1]*10)

        # Additional reward for reaching the goal
        if obs_next[0] >= 0.5:
            shaped_reward += 100 # Large bonus for success


        # Convert to tensor
        reward_tensor = torch.tensor([shaped_reward], device=device)

        next_state = None if done else get_state(obs_next)
        memory.push(state, action, next_state, reward_tensor)

        state = next_state
        total_reward += reward # Keep track of the TRUE environment reward
        total_steps += 1

        if total_steps > initial_memory:
            optimize_model()

        if done:
            if obs_next[0] >= 0.5:
                successes += 1
            break

    rewards_list.append(total_reward)

    if i_episode >= mean_n:
        mean_r = np.mean(rewards_list[-mean_n:])
        mean_rewards.append(mean_r)
        best_mean = max(best_mean, mean_r)
        best_mean_rewards.append(best_mean)
        steps_list.append(total_steps)

    if i_episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {i_episode+1}/{num_episodes} | Reward: {total_reward:.2f} | "
          f"Success: {successes} | epsilon={eps_threshold:.3f}")

###############
# Plotting
###############
env.close()

file_name = f"{args.environment}_DQN_{num_episodes}_episodes_{args.batch}batch.png"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Performance Plot
# Ensure steps_list and mean_rewards are not empty
if steps_list and mean_rewards:
    ax1.plot(steps_list, mean_rewards, label=f"{mean_n}-episode mean")
    ax1.plot(steps_list, best_mean_rewards, label="Best mean reward")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Reward")
ax1.set_title("DQN Performance on MountainCar-v0")
ax1.grid(True)
ax1.legend()


X = np.linspace(-1.5, 0.6, 50)
Y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(X, Y)
X, Y = X.flatten(), Y.flatten()


Z = []
# Using policy_net to determine the action for each state
states_to_evaluate = torch.tensor(np.array(list(zip(X, Y))), dtype=torch.float32, device=device)
with torch.no_grad():
    actions_tensor = policy_net(states_to_evaluate).argmax(dim=1).cpu().numpy() # Get index of max Q-value
Z = actions_tensor

colors = ['lime', 'red', 'blue'] # 0: left, 1: no-op, 2: right
ax2.scatter(X, Y, c=[colors[z] for z in Z], s=1, alpha=0.7)
ax2.set_xlabel("Position")
ax2.set_ylabel("Velocity")
ax2.set_title("Trained DQN Action Choices")
legend_recs = [mpatches.Patch(color=colors[i], label=f"Action {i}") for i in range(3)]
ax2.legend(handles=legend_recs)
ax2.set_xlim([-1.5, 0.6])
ax2.set_ylim([-1, 1])


plt.tight_layout()
plt.savefig(file_name, dpi=200)
print(f"Saved plot to {file_name}")
