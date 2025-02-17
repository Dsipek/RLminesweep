import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from dqn_model import CNN
from replay_buffer import ReplayBuffer
from config import learn_rate, DISCOUNT, epsilon, EPSILON_DECAY, EPSILON_MIN, UPDATE_TARGET_EVERY

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = DISCOUNT
        self.epsilon = epsilon
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.update_target_every = UPDATE_TARGET_EVERY

        self.memory = ReplayBuffer()
        self.model = CNN((1, env.size, env.size), env.size * env.size)
        self.target_model = CNN((1, env.size, env.size), env.size * env.size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learn_rate)
        self.learn_step_counter = 0
        self.loss = 0

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

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
            next_q_values = self.target_model(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss = loss.item()
        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
