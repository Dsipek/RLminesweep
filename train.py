import matplotlib.pyplot as plt
from minesweep_env import MinesweeperEnv
from dqn_agent import DQNAgent
from config import BATCH_SIZE, LEARN_DECAY, LEARN_MIN

env = MinesweeperEnv(size=9, n_mines=8)
agent = DQNAgent(env)
num_episodes = 1000
losses = []

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.set_title("Loss over Time")

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
        agent.learn(BATCH_SIZE)
        losses.append(agent.loss)

        if agent.learn_step_counter % agent.update_target_every == 0:
            agent.update_target_network()
        
        if env.is_win():
            print(f"Episode {episode + 1}: Total Reward = {total_reward} - WIN!")
            break

    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
    agent.optimizer.param_groups[0]['lr'] = max(agent.optimizer.param_groups[0]['lr'] * LEARN_DECAY, LEARN_MIN)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}, Loss: {agent.loss}")

    if episode % 50 == 0:
        ax.clear()
        ax.plot(losses, label="Loss")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss over Time")
        ax.legend()
        plt.draw()
        plt.pause(0.001)    # Small pause to ensure the plot updates
        plt.savefig("loss.png")  # Save current plot to a PNG file

plt.ioff()  # Turn off interactive mode
plt.show()