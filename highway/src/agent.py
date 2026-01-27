import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from src.config import Config

# Sinir Ağı Mimarisi
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_SIZE, Config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_SIZE, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Ajan Sınıfı
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_net = DQN(state_dim, action_dim).to(Config.DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(Config.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LR)
        self.memory = deque(maxlen=Config.MEMORY_CAPACITY)
        self.epsilon = Config.EPSILON_START

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(Config.DEVICE)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < Config.BATCH_SIZE:
            return

        batch = random.sample(self.memory, Config.BATCH_SIZE)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(Config.DEVICE)
        action = torch.LongTensor(action).unsqueeze(1).to(Config.DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(Config.DEVICE)
        next_state = torch.FloatTensor(np.array(next_state)).to(Config.DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(Config.DEVICE)

        current_q = self.policy_net(state).gather(1, action)
        
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (Config.GAMMA * next_q * (1 - done))

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=Config.DEVICE))