import gym
import torch
import os
from datetime import datetime
import numpy as np
from ppo_agent import PPOAgent

def train(agent, env, episodes=1000, max_steps=800, update_interval=2000):
    episode_rewards = []

    states = []
    actions = []
    old_probs = []
    rewards = []
    dones = []

    step = 0
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            probs, _ = agent.policy(torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0))
            action_prob = probs.squeeze()[action]
            states.append(state)
            actions.append(action)
            old_probs.append(action_prob.item())
            rewards.append(reward)
            dones.append(done)

            state = next_state
            episode_reward += reward
            step += 1

            if step % update_interval == 0:
                agent.update(np.array(states), np.array(actions), np.array(old_probs), np.array(rewards), np.array(dones))
                states, actions, old_probs, rewards, dones = [], [], [], [], []

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode}/{episodes}, Reward: {episode_reward}")

    return episode_rewards

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = PPOAgent(state_size, action_size)

    episode_rewards = train(agent, env)

    # Save the trained model
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"ppo_lunar_lander_{now}.pth")
    agent.save_model(model_path)

    # Plot the rewards
    import matplotlib.pyplot as plt
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Save the figure to a file
    plt.savefig("ppo_training_rewards.png", dpi=300)

    # Show the plot
    plt.show()


