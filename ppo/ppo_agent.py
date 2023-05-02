import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class PPOAgent:
    def __init__(self, state_size, action_size, lr=0.01, gamma=0.99, epsilon_clip=0.1, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip

        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.policy(state)
        return torch.multinomial(probs, num_samples=1).item()

    def compute_loss(self, states, actions, old_probs, rewards, dones):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_probs = torch.tensor(old_probs, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        probs, values = self.policy(states)
        action_probs = probs.gather(1, actions.unsqueeze(-1)).squeeze()

        advantages = rewards - values.squeeze()
        critic_loss = advantages.pow(2).mean()

        ratios = action_probs / old_probs
        surr1 = ratios * advantages.detach()
        surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages.detach()
        actor_loss = -torch.min(surr1, surr2).mean()

        return actor_loss + 0.5 * critic_loss

    def normalize_rewards(self, rewards):
        rewards = np.array(rewards)
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-10  # Add a small constant to avoid division by zero
        normalized_rewards = (rewards - mean) / std
        return normalized_rewards

    def update(self, states, actions, old_probs, rewards, dones):
        normalized_rewards = self.normalize_rewards(rewards)
        loss = self.compute_loss(states, actions, old_probs, normalized_rewards, dones)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))
