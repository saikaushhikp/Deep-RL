# Part(a)

A random agent in LunarLander-v3 consistently receives low, negative rewards with fluctuating values, showing that random actions just lead to frequent crashes or failed landings

---

![Random Agent Performance](part(a)/LunarLander-v3_50_16_random_agent.png)
<center>Figure-1: Rewards per episode for a random agent in LunarLander-v3 environment over 50 episodes
</center>

---

---

# Part(b)

- Reward-to-Go (RTG) makes the agent learn faster, hitting higher returns quickly and avoiding those annoying plateaus that total-return methods can get stuck in.

- Advantage Normalization (AdvNorm) smooths out the learning curve, cutting down on instability and making the whole process more reliable, so the agent gets better results overall.

- RTG + AdvNorm is the best-it leads to fast learning, steady performance, and consistently high returns (200+).

- RTG without AdvNorm still does okay, but it's a little less stable, with some dips and slower recovery compared to when both methods are combined.

- No RTG method + without advantage normalization are the least stable-they often regress mid-training and struggle with learning.

---

![rTaT](part(b)/LunarLander-v3_PG_iters1200_bs800_g_lr_rtg_advnorm_.png)
![rTaF](part(b)/LunarLander-v3_PG_iters1200_bs800_g_lr_rtg_noadv_.png)
![rFaT](part(b)/LunarLander-v3_PG_iters1200_bs800_g_lr_tot_advnorm_.png)
![rFaF](part(b)/LunarLander-v3_PG_iters1200_bs800_g_lr_tot_noadv_.png)
<center>Figure-2,3,4,5: Average Return over 1200 episodes for 800 as batch size with different combinations of Reward-to-Go and Advantage Normalization
</center>

--- 

---


# Part(c)

- Small Batch (bs=80) is bumpy rewards-lots of noise, so the agent barely makes progress. It’s basically learning from unstable, messy updates. And the max return hovers around 0 (1200 episodes).

- Medium Batch (bs=800) smooths things out-less noise, faster learning, and the agent hits around 200 returns. 

- Large Batch (bs=8000) is the smoothest ride with high, consistent returns (250-300), but it’s slower to converge because the agent updates less often.

- Bigger batches = smoother, slower, and more stable. Smaller batches = faster, risky convergence.

---

![bs80](part(c)/LunarLander-v3_PG_iters1200_bs80_g_lr_rtg_advnorm_.png)
![bs800](part(c)/LunarLander-v3_PG_iters1200_bs800_g_lr_rtg_advnorm_.png)
![bs8000](part(c)/LunarLander-v3_PG_iters1200_bs8000_g_lr_rtg_advnorm_.png)
<center>Figure-6,7,8: Average Return over 1200 episodes for different batch sizes with Reward-to-Go and Advantage Normalization enabled
</center>

---

---