from minesweep_protocol import MinesweeperStateProtocol
from typing import Tuple
import numpy as np

class RunningState(MinesweeperStateProtocol):
    """
    Represents the state of the game when the game is running.
    """

    def step(self, env, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        i,j = action

        if env.revealed[i,j]:
            return env.get_state(), -0.5, False, {}
        
        env.revealed[i ,j] = True

        if env.board[i, j] == -1:
            env.done = True
            env.current_state = TerminalState()
            return env.get_state(), -10, True, {}
        
        reward = 1
        
        if np.sum(env.revealed) == env.size * env.size - env.n_mines:
            env.done = True
            reward += 10
            env.current_state = TerminalState()
            return env.get_state(), reward, True, {}
        
        return env.get_state(), reward, False, {}

class TerminalState(MinesweeperStateProtocol):
    """
    Represents the state where the game is over
    """
    def step(self, env, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        return env.get_state(), 0, True, {}

