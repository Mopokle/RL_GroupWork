import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random

# Hyperparameters
ALPHA = 0.2
TAU = 0.005
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
REPLAY_BUFFER_SIZE = 1000000



# Save and load paths
MODEL_SAVE_PATH = "sac_lunar_lander_model.pth"
VIDEO_SAVE_PATH = "video"

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done)

    def __len__(self):
        return len(self.buffer)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample(self, state):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound):
        self.value_net = ValueNetwork(state_dim).cuda()
        self.target_value_net = ValueNetwork(state_dim).cuda()
        self.policy_net = PolicyNetwork(state_dim, action_dim, action_bound).cuda()
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=LEARNING_RATE)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.update_target_network()

    def update_target_network(self):
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(BATCH_SIZE)

        state = torch.FloatTensor(state).cuda()
        action = torch.FloatTensor(action).cuda()
        reward = torch.FloatTensor(reward).unsqueeze(1).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        done = torch.FloatTensor(done).unsqueeze(1).cuda()

        with torch.no_grad():
            next_action, next_log_prob = self.policy_net.sample(next_state)
            next_q_value = self.target_value_net(next_state)
            target_q_value = reward + GAMMA * (1 - done) * (next_q_value - ALPHA * next_log_prob)

        q_value = self.value_net(state)
        value_loss = (q_value - target_q_value.detach()).pow(2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        sampled_action, log_prob = self.policy_net.sample(state)
        q_value = self.value_net(state)
        policy_loss = (ALPHA * log_prob - q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.update_target_network()

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        action, _ = self.policy_net.sample(state)
        return action.item()

    def save_model(self, path):
        checkpoint = {
            "value_net": self.value_net.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "target_value_net": self.target_value_net.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_value_net.load_state_dict(checkpoint["target_value_net"])
