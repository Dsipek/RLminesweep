from minesweep_env import MinesweeperEnv
from dqn_agent import DQNAgent
import torch

env = MinesweeperEnv(size=5, n_mines=3)
agent = DQNAgent(env) #, lr=0.0001)
num_episodes = 1000
batch_size = 32

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.learn(batch_size)
        
        if env.is_win():
            print(f"Episode {episode + 1}: Total Reward = {total_reward} - WIN!")
            break

    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")
