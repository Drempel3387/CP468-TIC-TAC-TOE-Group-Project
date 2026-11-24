"""Core Tic-Tac-Toe game helpers and game-loop orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .board import Board


@dataclass
class GameResult:
    winner: Optional[str]
    final_state: Board
    moves: list[int]

    @property
    def is_draw(self) -> bool:
        return self.winner is None


def play_game(agent_x, agent_o, size: int = 3, win_length: Optional[int] = None,
              stochastic: bool = False, rng=None, display=None) -> GameResult:
    """Play agents against each other and return the result.

    Each agent must implement ``choose_move(state)``. ``display`` may be a
    callback receiving the state after each move.
    """
    from .stochastic import resolve_stochastic_move
    import random

    rng = rng or random.Random()
    state = Board.empty(size, win_length)
    agents = {"X": agent_x, "O": agent_o}
    history: list[int] = []
    if display:
        display(state)
    while not state.is_terminal():
        player = state.current_player
        intended = agents[player].choose_move(state)
        if intended not in state.get_legal_moves():
            raise ValueError(f"Agent {player} returned illegal move {intended!r}.")
        actual = resolve_stochastic_move(state, intended, rng=rng) if stochastic else intended
        state = state.apply_move(actual)
        history.append(actual)
        if display:
            display(state)
    return GameResult(state.winner(), state, history)
