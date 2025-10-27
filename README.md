# Deep Reinforcement Learning: DQN & Policy Gradient

This repository contains implementations of **Deep Q-Network (DQN)** and **Policy Gradient (PG)** algorithms. The goal of this project is to apply these reinforcement learning techniques on various **OpenAI Gym environments** and analyse their performance, learning behaviour, and hyperparameter sensitivities.

---

## Overview

In this assignment, I implement two fundamental reinforcement learning algorithms:

1. **Deep Q-Network (DQN)** - A value-based method that uses deep neural networks to approximate the Q-function.
2. **Policy Gradient (PG)** - A policy-based method that directly optimises the policy by gradient ascent on expected rewards.

The algorithms are tested on multiple OpenAI Gym environments, ranging from simple control tasks to more challenging Atari games like Pong.

---

## Objectives

- Implement and train **DQN** and **Policy Gradient** algorithms from scratch.  
- Evaluate their performance on various **Gym environments**.  
- Analyse the effect of different **hyperparameters** (learning rate, exploration schedule, etc.).  
- Visualise training progress using **learning curves**.  
- Explore additional environments for deeper insights or bonus experiments.

---

## Tested Environments

| Algorithm | Environment Examples | Complexity |
|------------|----------------------|-------------|
| DQN | `CartPole-v1`, `MountainCar-v0`, `Pong-v0` | Simple → Medium |
| Policy Gradient | `CartPole-v1`, `Acrobot-v1`, `InvertedPendulum-v2` | Simple → Continuous |

> ⚠️ **Note:** Training Pong with DQN can take a long time - up to 2-3 nights on a modest laptop. Plan your experiments accordingly.


