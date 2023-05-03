LunarLander RL Group Project 🚀🌕
=================================

This is a group project for learning Reinforcement Learning using OpenAI Gym's LunarLander environment. The objective of the project is to train an agent to land a lunar module safely on the moon. We experiment with two popular RL algorithms, Deep Q-Network (DQN) and Proximal Policy Optimization (PPO), and compare their performance.

--------
![Alt Text](<img src="https://github.com/Mopokle/RL_GroupWork/raw/main/lunargif.gif" width="500"/>
)
--------
Table of Contents 📚
--------------------

*   [Getting Started](#getting-started)
*   [Prerequisites](#prerequisites)
*   [Installation](#installation)
*   [Training the Agents](#training-the-agents)
*   [Testing the Agents](#testing-the-agents)
*   [Built With](#built-with)
*   [Contributors](#contributors)

Getting Started 🌟
------------------

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites 📝

*   Python 3.7 or higher
*   PyTorch
*   OpenAI Gym

### Installation 🛠️

1.  Clone the repository:

```bash
git clone https://github.com/Mopokle/RL_GroupWork.git
```

2.  Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # Linux or macOS
venv\Scripts\activate     # Windows
```

3.  Install the required packages:

```bash
pip install -r requirements.txt
```

Training the Agents 🎓
----------------------

You can train the agents using DQN and PPO algorithms. 

To train the DQN agent, navigate to the `dqn` folder and run:

```bash
python dqn_train.py
```

To train the PPO agent, navigate to the `ppo` folder and run:

```bash
python ppo_train.py
```

The training scripts will display the rewards and save the trained models to the respective folders.

Testing the Agents 🕹️
----------------------

To test the trained agents, you can run the following commands:

For the DQN agent:

```bash
python dqn_test.py
```

For the PPO agent:

```bash
python ppo_test.py
```

These scripts will load the trained models and visualize the agents' performance in the LunarLander environment.

Built With 🛠️
--------------

*   [Python](https://www.python.org/) - The programming language used
*   [PyTorch](https://pytorch.org/) - The deep learning framework used
*   [OpenAI Gym](https://gym.openai.com/) - Toolkit for developing and comparing reinforcement learning algorithms

Group Members 👥
---------------

*   [Lingfeng Wang](https://github.com/Mopokle)
*   [Jiacheng Xu](https://github.com/unfaa3)
*   [Yi-Lun Chu](https://github.com/chuyilun)
*   [Junjie Liu](https://github.com/wodigexiaodonggua)
*   [Chun Hung Lin](https://github.com/efpm168806)
*   [Zhechen Huang](https://github.com/JasonHuang0028)