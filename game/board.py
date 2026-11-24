"""Immutable N x N Tic-Tac-Toe board/state used by all search algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Board:
    """A game position. Moves are zero-based cell indices (0 .. size*size-1).

    By default, a player wins by filling an entire row, column, or diagonal.
    ``win_length`` can be set for variants such as connect-k on a larger board.
    """
    cells: tuple[Optional[str], ...]
    current_player: str = "X"
    size: int = 3
    win_length: Optional[int] = None

    def __post_init__(self):
        if self.size < 1 or len(self.cells) != self.size * self.size:
            raise ValueError("Board cells must contain exactly size * size entries.")
        if self.current_player not in ("X", "O"):
            raise ValueError("current_player must be 'X' or 'O'.")
        if any(cell not in (None, "X", "O") for cell in self.cells):
            raise ValueError("Cells must be None, 'X', or 'O'.")
        k = self.win_length if self.win_length is not None else self.size
        if not 1 <= k <= self.size:
            raise ValueError("win_length must be between 1 and board size.")
        object.__setattr__(self, "win_length", k)

    @classmethod
    def empty(cls, size: int = 3, win_length: Optional[int] = None) -> "Board":
        return cls((None,) * (size * size), "X", size, win_length)

    def get_legal_moves(self) -> list[int]:
        return [i for i, value in enumerate(self.cells) if value is None]

    def legal_moves(self) -> list[int]:
        return self.get_legal_moves()

    def apply_move(self, move: int) -> "Board":
        if not isinstance(move, int) or isinstance(move, bool):
            raise ValueError("Move must be an integer cell index.")
        if move < 0 or move >= len(self.cells):
            raise ValueError(f"Move must be between 0 and {len(self.cells) - 1}.")
        if self.cells[move] is not None:
            raise ValueError(f"Cell {move + 1} is already occupied.")
        updated = list(self.cells)
        updated[move] = self.current_player
        return Board(tuple(updated), other_player(self.current_player), self.size, self.win_length)

    def winner(self) -> Optional[str]:
        n, k = self.size, self.win_length
        # Check every possible start point in the four winning directions.
        for row in range(n):
            for col in range(n):
                mark = self.cells[row * n + col]
                if mark is None:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    end_r, end_c = row + (k - 1) * dr, col + (k - 1) * dc
                    if not (0 <= end_r < n and 0 <= end_c < n):
                        continue
                    if all(self.cells[(row + step * dr) * n + col + step * dc] == mark
                           for step in range(k)):
                        return mark
        return None

    def is_draw(self) -> bool:
        return self.winner() is None and all(cell is not None for cell in self.cells)

    def is_terminal(self) -> bool:
        return self.winner() is not None or self.is_draw()

    def utility(self, player: str) -> int:
        winner = self.winner()
        return 0 if winner is None else (1 if winner == player else -1)

    def __str__(self) -> str:
        width = len(str(self.size * self.size))
        rows = []
        for r in range(self.size):
            rows.append(" | ".join(
                (self.cells[r * self.size + c] or str(r * self.size + c + 1)).rjust(width)
                for c in range(self.size)
            ))
        separator = "-+-".join("-" * width for _ in range(self.size))
        return ("\n" + separator + "\n").join(rows)


def other_player(player: str) -> str:
    if player not in ("X", "O"):
        raise ValueError("Player must be 'X' or 'O'.")
    return "O" if player == "X" else "X"


# Descriptive alias for code that prefers an explicit state name.
TicTacToeState = Board
GameState = Board
