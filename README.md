# Reinforcement Learning on Gymnasium Classic Control Environments using Ray RLlib

This project implements reinforcement learning (RL) algorithms using Ray RLlib on Gymnasium's classic control environments. It includes custom modifications to randomize the initial states for more robust policy training and evaluation.

## Algorithms

- **PPO** – Proximal Policy Optimization  
- **DQN** – Deep Q-Network  
- **SAC** – Soft Actor-Critic  
- **A2C** – Advantage Actor-Critic  
- **PG** – Policy Gradient

## Environments

- `CartPole-v1`  
- `MountainCar-v0`  
- `MountainCarContinuous-v0`  
- `Pendulum-v1`  
- `Acrobot-v1`

## Features

- Randomized initial states for each environment
- CPU and memory monitoring during training
- Final policy rollout and state trajectory visualization
- All agents trained for 300 iterations (or early-stopped if converged)

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate avis
```

## How to Run

Each environment has its own subdirectory with training scripts per algorithm. Example:

```bash
cd cartpole/PPO
python ppo.py
```

All the logs, checkpoints,rollout plots, and inference results are saved automatically and their paths are shown in the logs during the runtime.