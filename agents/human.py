"""Human-controlled player for the command-line interface."""
from .base import Agent


class HumanAgent(Agent):
    name = "Human"

    def __init__(self, input_fn=input, output_fn=print):
        self.input_fn = input_fn
        self.output_fn = output_fn

    def choose_move(self, state) -> int:
        legal = state.get_legal_moves()
        while True:
            raw = self.input_fn(f"Player {state.current_player}, choose a square (1-{state.size * state.size}): ").strip()
            try:
                move = int(raw) - 1
            except ValueError:
                self.output_fn("Please enter a valid square number.")
                continue
            if move not in legal:
                self.output_fn("That square is unavailable. Choose an empty square.")
                continue
            return move
