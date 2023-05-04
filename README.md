LunarLander RL Group Project 🚀🌕
=================================

This is a group project for learning Reinforcement Learning using OpenAI Gym's LunarLander environment. The objective of the project is to train an agent to land a lunar module safely on the moon. We experiment with four popular RL algorithms, Deep Q-Network (DQN), Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), and Advantage Actor-Critic (A2C), and compare their performance.


<figure>
  <p align="center">
    <img src="https://github.com/Mopokle/RL_GroupWork/raw/main/lunargif.gif" width="500"/>
  </p>
  <figcaption>Result</figcaption>
</figure>


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

You can train the agents using DQN, PPO, SAC, and A2C algorithms. To train the agents, navigate to the respective folders and run the corresponding training script:


```bash
cd "replace with foldername"
python "replace with algorithms"_train.py
```

The training scripts will display the rewards and save the trained models to the respective folders.

Testing the Agents 🕹️
----------------------

To test the trained agents, navigate to the respective folders and run the corresponding test script:

```bash
python "replace with algorithms"_test.py
```


The training scripts will display the rewards and save the trained models to the respective folders.

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


Project Documentation 📄
------------------------

For more details on the project, including discussions, meeting notes, and other resources, please visit our [Notion page](https://www.notion.so/mopokle/RL-Group-Project-9cfb2fcbd34048b582a35bb889f67664).