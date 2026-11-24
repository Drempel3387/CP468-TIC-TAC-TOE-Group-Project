"""Common interface for human and AI players."""
from abc import ABC, abstractmethod


class Agent(ABC):
    name = "Agent"

    @abstractmethod
    def choose_move(self, state) -> int:
        """Return a legal, zero-based move index for the supplied state."""
        raise NotImplementedError

    def __str__(self):
        return self.name
