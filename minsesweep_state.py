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

        if env.board[i, j] == 9:
            env.done = True
            env.current_state = TerminalState()
            return env.get_state(), -100, True, {}
        
        if env.is_win():
            env.done = True
            env.current_state = TerminalState()
            return env.get_state(), 1000, True, {}
        
        return env.get_state(), 1, False, {}

class TerminalState(MinesweeperStateProtocol):
    """
    Represents the state where the game is over
    """
    def step(self, env, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        return env.get_state(), 0, True, {}
