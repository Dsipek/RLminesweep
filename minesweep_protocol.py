from typing import Protocol
from typing import Tuple
import numpy as np

class MinesweeperStateProtocol(Protocol):
    """
    Defines the interface for Minesweeper state classes.
    Any state implementing this protocol must define a step method
    """
    def step(self, env: "MinesweeperEnv", action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, dict]:
        ...

