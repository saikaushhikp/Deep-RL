## Part(a)
A random agent in the MountainCar-v0 environment performs poorly, consistently receiving rewards around -200 without any improvement over time. Its actions are uniformly random across all states, which is an ineffective strategy for building the momentum required to escape the valley and reach the goal.

--- 

![Random Agent Performance and Policy Map](part(a)/MountainCar-v0_random_100ep_mean5.png)
<center>
Figure-1: Random Agent Performance (left) and its Policy Map (right) for 100 episodes of random actions.
</center>

---

---


# Part(b)
The DQN agent successfully learned to solve the MountainCar-v0 environment. This is demonstrated by three key outcomes:

- Training Loss: The loss initially rose and then steadily decreased, indicating successful convergence of the Q-network.
- Reward Improvement: The agent's average reward consistently increased throughout training, showing it learned to perform better.
- Effective Policy: The final policy map reveals the agent learned the essential momentum strategy, choosing actions that build momentum to reach the goal.

--- 

![DQN Agent Performance and Policy Map](part(b)/MountainCar-v0_DQN_1000_episodes_64batch.png)
<center>
Figure-2: DQN Agent Performance (left) and its Policy Map (right) after training for 1000 episodes with a batch size of 64.
</center>

---

![Agent's Training Loss](part(b)/MountainCar-v0_DQN_1000_episodes_64batch_loss.png)
<center>
Figure-3: DQN Agent Training Loss over 1000 episodes with a batch size of 64.
</center>

---

---


# Part(c)
- Small batch sizes yield greater update randomness and learning curve variance, potentially helping escape local minima but also risking instability.
- Large batch sizes offer steadier convergence and cleaner policy maps, but is observed that that 'No action' option isn't learned as effectively with larger batches.
- For MountainCar-v0, all tested batch sizes enable successful DQN convergence.

---

![](part(c)/MountainCar-v0_DQN_1000_episodes_16batch.png)
![](part(c)/MountainCar-v0_DQN_1000_episodes_32batch.png)
![](part(c)/MountainCar-v0_DQN_1000_episodes_64batch.png)
![](part(c)/MountainCar-v0_DQN_1000_episodes_128batch.png)
<center>
Figures 4,5,6,7: DQN Agent Performance for batch sizes of 16, 32, 64, and 128 over 1000 episodes.
</center>

---

---
