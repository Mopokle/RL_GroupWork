import gym
import numpy as np
import os
from datetime import datetime
from dqn_agent import DQNAgent

def train(agent, env, episodes=1000, max_steps=800, update_frequency=10, epsilon_start=1.0, epsilon_decay=0.998, epsilon_min=0.01):
    episode_rewards = []
    epsilon = epsilon_start

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.buffer) >= agent.batch_size:
                agent.update()

            if done:
                break

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if episode % update_frequency == 0:
            agent.update_target_network()
            print(f"Episode {episode}/{episodes}, Reward: {total_reward}, Epsilon: {epsilon}")

    return episode_rewards

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    episode_rewards = train(agent, env)

    # Save the trained model with the current time in the model folder
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"dqn_lunar_lander_{now}.pth")
    agent.save_model(model_path)

    # Plot the rewards
    import matplotlib.pyplot as plt
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # Save the figure to a file
    plt.savefig("training_rewards.png", dpi=300)

    plt.show()
