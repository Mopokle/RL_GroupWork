import gym
import time
import os
from dqn_agent import DQNAgent
from gym.wrappers import Monitor

def test(agent, env, episodes=10, max_steps=1000, render=True, sleep_time=0.01, video_file=None):
    if video_file is not None:
        env = Monitor(env, video_file, force=True, video_callable=lambda episode_id: True)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state, epsilon=0)  # Always use the greedy action for testing
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if render:
                env.render()
                time.sleep(sleep_time)

            if done:
                break

        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

    if video_file is not None:
        env.close()

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Load the trained model from the model folder
    model_dir = "model"
    model_file = "dqn_lunar_lander_2023-05-03_15-28-39.pth"
    model_path = os.path.join(model_dir, model_file)
    agent.load_model(model_path)

    # Set the output video file name
    video_file = "videos"

    # Test the model
    test(agent, env, video_file=video_file)

    env.close()
