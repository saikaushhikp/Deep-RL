
import subprocess
import os
import sys

# Fixed settings
environment = "CartPole-v1"
iterations = 500
learning_rate = 1e-3
gamma = 0.99

# Batch sizes to compare
batch_sizes = [50,500,5000]

# Store results
batch_rewards = {}

for b in batch_sizes:
    script_path = os.path.join(os.path.dirname(__file__), "CartPole-PG-b.py")
    try:
        result = subprocess.run(
            [
                sys.executable,
                script_path,
                "--environment", environment,
                "--iterations", str(iterations),
                "--batch_size", str(b),
                "--lr", str(learning_rate),
                "--gamma", str(gamma),
                "--reward_to_go",
                "--advantage_norm"
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(__file__)
        )
        # show subprocess output for debugging / parsing
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("Subprocess stderr:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Subprocess failed with returncode", e.returncode)
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        continue
