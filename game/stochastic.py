"""Chance mechanics for the stochastic Tic-Tac-Toe variant."""
from __future__ import annotations
import random

DEFAULT_INTENDED_PROBABILITY = 0.90


def move_outcomes(state, intended_move: int, intended_probability: float = DEFAULT_INTENDED_PROBABILITY):
    """Return ``(actual_move, probability)`` outcomes for an intended move.

    With probability p the selected square is used. With probability 1-p,
    the move lands uniformly on one of the *other* currently empty squares.
    """
    if not 0.0 <= intended_probability <= 1.0:
        raise ValueError("intended_probability must be between 0 and 1.")
    legal = state.get_legal_moves()
    if intended_move not in legal:
        raise ValueError("The intended move must be legal.")
    alternatives = [move for move in legal if move != intended_move]
    outcomes = [(intended_move, intended_probability)]
    if alternatives and intended_probability < 1.0:
        probability_each = (1.0 - intended_probability) / len(alternatives)
        outcomes.extend((move, probability_each) for move in alternatives)
    else:
        # With only one empty cell, the move necessarily lands there.
        outcomes = [(intended_move, 1.0)]
    return outcomes


def resolve_stochastic_move(state, intended_move: int, rng=None,
                            intended_probability: float = DEFAULT_INTENDED_PROBABILITY) -> int:
    """Sample the actual square resulting from an intended move."""
    rng = rng or random
    outcomes = move_outcomes(state, intended_move, intended_probability)
    roll = rng.random()
    cumulative = 0.0
    for move, probability in outcomes:
        cumulative += probability
        if roll < cumulative:
            return move
    return outcomes[-1][0]  # floating-point safety
