
import subprocess
import sys
import os

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