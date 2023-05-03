import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import os
from datetime import datetime
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from a2c_agent import A2CAgent

def train(agent, env, episodes=2000, max_steps=800):
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        trajectory = []

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if done:
                break

        agent.update(trajectory)
        episode_rewards.append(total_reward)
        print(f"Episode {episode}/{episodes}, Reward: {total_reward}")

    return episode_rewards

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = A2CAgent(state_size, action_size)

    episode_rewards = train(agent, env)

    # Save the trained model with the current time in the model folder
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"a2c_lunar_lander_{now}.pth")
    agent.save_model(model_path)

    # Plot the rewards
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Save the figure to a file
    plt.savefig("training_rewards.png", dpi=300)

    plt.show()
