# Part(a)

- The random agent just keeps getting worse, losing more and more over both short (5) and longer (15) episodes in Pong-v5. No sign of improvement.

- Flat Rewards with Little Variation: The rewards look predictable and follow a stair-step pattern (since Pong rewards are either -1 or +1 according to atari games documentation). The random agent mostly ends up with negative rewards, and there’s not much change from one episode to the next.

- Mostly Negative Scores: At the end of each episode, the agent’s cumulative score is usually between -15 and -21. $\to$ it’s bad no matter the length of the episode.

---
![5ep Random Agent](part(a)/pong_5_random_agent_rewards.png)
![15ep Random Agent](part(a)/pong_15_random_agent_rewards.png)

<center>Figure-1: Rewards obtained by a random agent in Pong over 5(top) and 15(bottom) episodes.(indefinite episode lengths)</center>

---

---



# Part(b)

**Note :** 

> The trained DQN had checkpoints saved for every 0.25M steps up to 2.5M steps.  
During the 1st run, the agent was trained for 1M steps and it's performance was plotted as below, where the best mean reward until then was -10.22 for a running average of 100 episodes.

![1Msteps DQN Agent](part(b)/dqn_pong_learning_curves_1M.png)
<center>
Figure 2: Learning curve of DQN agent trained for 1M steps in Pong.  <br>
Avg-return curve(top), Training loss curve(bottom).
</center>

---

> After that, due to computational constraints, the agent had to train for each checkpoint seperately for the next 1.5M steps (i.e from 1M to 2.5M steps). The model weights and training states are stored at [dqn_pong_step_25L.pth](part(b)/dqn_pong_step_25L.pth) and [dqn_pong_final.pth](part(b)/dqn_pong_final.pth) (both are same to be honest).  
The issue here is, the training has been done, but the learning curves could not be recorded because the training for each checkpoint was done seperately and not in a sinlgle run for many steps(like in 1M).  

> So, the model was trained for anothe 0.1M steps(from 2.5M to 2.6M) to get the learning curves as shown below. The best mean reward until then was 7 for a running average for 10 episodes for a total of 40 new episodes. hence the learning curve for this 0.1M steps shown below is only for reference to show the improvement in performance after 2.5M steps.

> To further evaluate the performance of the DQN agent, it was tested for 100 episodes and the rewards obtained are shown in the file [evaluation.txt](part(b)/evaluation.txt). The agent performs significantly better than the random agent, winning most of the episodes with positive rewards.  
The final model results are saved @ [dqn_pong_results.npz](part(b)/dqn_pong_results.npz).

Thus the trainined model performed good and can do better for many more training steps(to 8M to touch max reward almost as close as 21.0) starting from the latest checkpoint(of 2.5M steps)

---

![0.1Msteps DQN Agent](part(b)/dqn_pong_learning_curves_26L.png)
<center>
Figure-3: Learning curve of DQN agent trained for an additional 0.1M steps in Pong(after 2.5M steps).  <br>
Avg-return curve(top), Training loss curve(bottom). <br>
Parameters : (lr, gamma) = (0.0001, 0.99), total training steps = 2.5M steps + ReplyBufferSize = 1M 
</center>

---

---


# Part(c)

## Unfortunate training contraints couldn't be met for doing part(c) & as adviced by sir in the Problem-statement
