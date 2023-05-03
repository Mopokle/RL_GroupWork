import gym
import time
import os
from ppo_agent import PPOAgent
from gym.wrappers import Monitor

def test(agent, env, episodes=10, max_steps=1000, render=True):
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)

            if render:
                env.render()
                time.sleep(0.01)

            episode_reward += reward

            if done:
                break

        print(f"Episode {episode}/{episodes}, Reward: {episode_reward}")

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = PPOAgent(state_size, action_size)

    model_dir = "model"
    model_file = "ppo_lunar_lander_ep2000.pth"
    model_path = os.path.join(model_dir, model_file)
    agent.load_model(model_path)

    # Wrap the environment with the Monitor
    video_output_dir = "videos"
    env = Monitor(env, video_output_dir, force=True, video_callable=lambda episode_id: True)

    test(agent, env)
    env.close()

