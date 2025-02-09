import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from dqn_model import CNN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, env, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99

        self.memory = ReplayBuffer(10000)
        self.model = CNN(env.size, env.size * env.size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)

    def select_action(self, state):
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.env.size * self.env.size - 1)
            row = action_index // self.env.size
            col = action_index % self.env.size
            return (row, col)
        # state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        action_index = torch.argmax(q_values).item()
        row = action_index // self.env.size
        col = action_index % self.env.size
        return (row, col)

    def learn(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
