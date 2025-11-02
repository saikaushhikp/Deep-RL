
import subprocess
import sys
import os

# Fixed settings
environment = "MountainCar-v0"
num_episodes = 1000


batch_sizes = [16, 32, 64, 128]

batch_rewards = {}

for b in batch_sizes:
    print(f"\nRunning with batch size = {b}")
    result = subprocess.run(
        [
            sys.executable,
            "MountainCar-DQN-b.py",
            "--environment", environment,
            "--num_episodes", str(num_episodes),
            "--batch_size", str(b),
        ],
        capture_output=True, text=True
    )

    batch_rewards[b] = result.stdout
    print("-"*10)
    print(result.stdout)
    if result.returncode != 0:
        print("--- STDERR (error) ---")
        print(result.stderr)