"""Agent that chooses moves using expectiminimax for stochastic play."""
from algorithms.expectiminimax import expectiminimax_search
from .base import Agent


class ExpectiminimaxAgent(Agent):
    name = "Expectiminimax"

    def __init__(self, intended_probability=0.90):
        self.intended_probability = intended_probability
        self.last_stats = None

    def choose_move(self, state):
        move, _value, stats = expectiminimax_search(
            state, state.current_player, self.intended_probability
        )
        self.last_stats = stats
        return move
