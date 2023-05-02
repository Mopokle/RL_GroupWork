# Lunar Lander Reinforcement Learning 🚀🌕

This repository contains a reinforcement learning project for training a lunar lander agent using Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms. The project is built using PyTorch and the OpenAI Gym's LunarLander-v2 environment.

## Table of Contents 📚

- [Overview](#overview)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Training and Testing](#training-and-testing)
- [Results](#results)
- [Contributing](#contributing)

## Overview 🔍

The goal of this project is to train an agent to successfully land the lunar module on the surface of the moon. The agent is trained using two popular reinforcement learning algorithms:

1. Deep Q-Network (DQN)
2. Proximal Policy Optimization (PPO)

Both DQN and PPO agents are implemented using PyTorch, and the training is conducted using the OpenAI Gym's LunarLander-v2 environment.

## Requirements 🛠️

- Python 3.6 or later
- [PyTorch](https://pytorch.org/get-started/locally/)
- [OpenAI Gym](https://gym.openai.com/docs/#installation)
- [NumPy](https://numpy.org/install/)

## Getting Started 🏁

1. Clone the repository:
```
git clone [https://github.com/Mopokle/RL_GroupWork 
```
2. Install the required packages:



```shell
pip install -r requirements.txt
```

## Training and Testing 🏋️‍♂️🔍

### DQN

To train the DQN agent:
```
python dqn_train.py
```
To test the DQN agent:
```
python dqn_test.py
```

### PPO
To train the PPO agent:
```shell
python ppo_train.py
```

To test the PPO agent:
```
python ppo_test.py
```

## Results 📊

The training progress and results are saved in the corresponding directories for each algorithm (DQN and PPO). Plots and trained models can be found in these directories.
