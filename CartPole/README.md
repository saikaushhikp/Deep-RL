# Part(a)

A random agent in the CartPole-v1 Gym environment receives consistently low and volatile rewards, usually between 15 and 28 per episode across many iterations, means the agent cannot learn or exploit the environment, leading to frequent failure in balancing the pole. Occasional higher rewards occur only by chance.

---

![Random Agent Performance](part(a)/CartPole-v1_500_16_random_agent.png)
<center>Figure-1: Rewards per episode for a random agent in CartPole-v1 environment over 500 <br>
Parameters:500 iterations, 16 timesteps per iteration('batch')
</center>

---

---

# Part(b)

- Advantage Normalization speeds up learning and makes it more stable, helping the model hit near-optimal performance pretty quickly.

- Reward-to-Go boosts performance too, but without advantage normalization, things get shaky, with some drops due to high variance.

- No Reward-to-Go leads to a bumpy ride, with performance dropping a lot, showing that it's less efficient and struggles with variance.

- Combining Both gives the best results, with faster learning, less fluctuation, and overall smoother training.

- Advantage Normalization on its own is the best for stabilizing things, especially in noisy environments like CartPole-v1.

- Without Either, you get a lot of instability, with performance going up and down, and it’s hard to maintain high returns.

---
![rTaT](part(b)/CartPole-v1_PG_iters500_bs5000_g_lr_rtg_advnorm_.png)
![rTaF](part(b)/CartPole-v1_PG_iters500_bs5000_g_lr_rtg_noadv_.png)
![rFaT](part(b)/CartPole-v1_PG_iters500_bs5000_g_lr_tot_advnorm_.png)
![rFaF](part(b)/CartPole-v1_PG_iters500_bs5000_g_lr_tot_noadv_.png)
<center>Figure-2,3,4,5: Average Return over 500 episodes for 5000 as batch size with different combinations of Reward-to-Go and Advantage Normalization <br>
Parameters : 500 iterations, 5000 timesteps per iteration('batch'), with learningrate = 0.001 , gamma = 0.99, reward_to_go and advantage_norm (booleans)
</center>

---

---

# Part(c)

- Small Batch (bs=50) is super noisy-lots of ups and downs, takes longer to stabilize, and even after a lot of training, the agent can still mess up.

- Medium Batch (bs=500) smooths things out a bit-faster and more reliable learning, but still some random dips later on.

- Large Batch (bs=5000) is the smoothest-consistent, stable, and hits optimal performance without big drops.

- Smaller batches update faster but with higher variance, making it more unstable in the long run time.

- Larger batches cut down on gradient noise, making the learning process steadier and more reliable.

- Big trade-off: small batches are faster but unstable, while large batches are slower but way more consistent.

---

![bs50](part(c)/CartPole-v1_PG_iters500_bs50_g_lr_rtg_advnorm_.png)
![bs500](part(c)/CartPole-v1_PG_iters500_bs500_g_lr_rtg_advnorm_.png)
![bs5000](part(c)/CartPole-v1_PG_iters500_bs5000_g_lr_rtg_advnorm_.png)
<center>Figure-6,7,8: Average Return over 500 episodes for different batch sizes with Reward-to-Go and Advantage Normalization enabled <br>
Parameters : 500 iterations, leaningrate = 0.001 , gamma = 0.99, reward_to_go and advantage_norm (booleans) , batch sizes 50,500,5000(experimented)
</center>

---

---
