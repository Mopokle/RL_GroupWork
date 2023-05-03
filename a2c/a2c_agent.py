import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import os
from datetime import datetime
import matplotlib.pyplot as plt
from torch.distributions import Categorical

class A2C(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(A2C, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-3, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.model = A2C(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        return int(action.item())

    def update(self, trajectory):
        states, actions, rewards, next_states, dones = zip(*trajectory)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        action_probs, state_values = self.model(states)
        _, next_state_values = self.model(next_states)
        distribution = Categorical(action_probs)

        action_log_probs = distribution.log_prob(actions)
        advantages = rewards + (1 - dones) * next_state_values.squeeze() - state_values.squeeze()
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        critic_loss = nn.functional.smooth_l1_loss(state_values.squeeze(), rewards + (1 - dones) * next_state_values.squeeze())
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
