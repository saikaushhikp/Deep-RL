
import subprocess
import matplotlib.pyplot as plt
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

# Fixed settings
environment = "MountainCar-v0"
num_episodes = 100

learning_rate = 1e-4
gamma = 0.9
mean_n = 5

# Batch sizes to compare
batch_sizes = [8, 16, 32, 64]

# Store results
batch_rewards = {}

for b in batch_sizes:
    print(f"\nRunning with batch size = {b}")
    result = subprocess.run(
        [
            "python3",
            "MountainCar-DQN-b.py",
            "--environment", environment,
            "--num_episodes", str(num_episodes),
            "--batch", str(b),
            "--learning_rate", str(learning_rate),
            "--gamma", str(gamma),
            "--mean_n", str(mean_n)
        ],
        capture_output=True, text=True
    )

#     Suppose your training script prints "Mean rewards per iteration: [...]"
#     Extract and parse that list (you can modify question3b.py to save or print it)
#     rewards = eval(result.stdout.split("Mean rewards per iteration: ")[-1])
#     batch_rewards[b] = rewards

# # Plot comparison
# plt.figure(figsize=(10, 6))
# for b, rewards in batch_rewards.items():
#     plt.plot(rewards, label=f'Batch Size = {b}')
# plt.title(f'Impact of Batch Size on Policy Gradient ({environment})')
# plt.xlabel('Iteration')
# plt.ylabel('Average Return')
# plt.legend()
# plt.grid()
# # plt.savefig(f'{environment}_batch_comparison.png')
# plt.show()
