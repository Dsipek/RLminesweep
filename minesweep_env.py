import random
import numpy as np
from typing import Tuple
from minsesweep_state import RunningState, TerminalState
from minesweep_protocol import MinesweeperStateProtocol

class MinesweeperEnv:
    """
    Environment for following the state machine
    """
    def __init__(self, size: int = 5, n_mines: int = 3):
        self.size = size
        self.n_mines = n_mines
        self.reset()

    def reset(self) -> MinesweeperStateProtocol:
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.revealed = np.full((self.size, self.size), False, dtype=bool)
        
        mines_placed = 0
        while mines_placed < self.n_mines:
            i = random.randint(0, self.size - 1)
            j = random.randint(0, self.size - 1)
            if self.board[i, j] == 0:
                self.board[i, j] = -1
                mines_placed += 1
            
        # Fill the numbers for safe cells. Counts of adjacent cells with mines
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i,j] != -1:
                    self.board[i,j] = self.count_adjacent_mines(i ,j)

        self.done = False
        self.current_state: MinesweeperStateProtocol = RunningState()
        return self.get_state()

    def count_adjacent_mines(self, i, j):
        """
        Counts the number of mines adjacent to the given cell
        """
        count = 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0 ,1]:
                ni, nj = i + di, j + dj
                if 0<= ni < self.size and 0 <= nj <self.size and self.board[ni, nj] == -1:
                    count += 1
        
        return count

    def get_state(self):
        """
        Returns current game state
        """
        state = np.full((self.size, self.size), -2, dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                if self.revealed[i, j]:
                    state[i, j] = self.board[i, j]
        
        return state

    def step(self, action: Tuple[int, int]) -> Tuple[MinesweeperStateProtocol, float, bool, dict]:
        """
        Processes an action by calling the current states step method 
        """
        return self.current_state.step(self, action)