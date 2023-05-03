import gym
import time
import os
from a2c_agent import A2CAgent
from gym.wrappers import Monitor

def test(agent, env, episodes=10, max_steps=1000, render=True, sleep_time=0.01, video_file=None):
    if video_file is not None:
        env = Monitor(env, video_file, force=True, video_callable=lambda episode_id: True)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)  # Always use the greedy action for testing
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if render:
                env.render()
                time.sleep(sleep_time)

            if done:
                break

        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = A2CAgent(state_size, action_size)

    # Load the trained model from the model folder
    model_dir = "model"
    model_file = "a2c_lunar_lander_ep2000.pth"  # Make sure to use the correct file name for the A3C model
    model_path = os.path.join(model_dir, model_file)
    agent.load_model(model_path)

    # Set the output video file name
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)  # Create the output video folder if it doesn't exist
    video_file = os.path.join(video_dir, "a2c_lunar_lander_test.mp4")

    # Test the model
    test(agent, env, video_file=video_file)

    env.close()
