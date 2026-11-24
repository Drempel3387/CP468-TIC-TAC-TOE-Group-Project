"""Random legal-move baseline agent."""
import random
from .base import Agent


class RandomAgent(Agent):
    name = "Random"

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose_move(self, state):
        moves = state.get_legal_moves()
        if not moves:
            raise ValueError("No legal moves remain.")
        return self.rng.choice(moves)
