"""Prompt construction for the Gemini Tic-Tac-Toe agent."""
import json


def move_prompt(state) -> str:
    """Build a concise prompt that constrains Gemini to a legal move."""
    board = [cell if cell is not None else "_" for cell in state.cells]
    return (
        "You are an expert Tic-Tac-Toe player. Choose one legal empty square. "
        "Return only a JSON object with one integer field named move. "
        "The move is a zero-based index in row-major order. Do not include markdown.\n"
        f"Board size: {state.size} x {state.size}\n"
        f"Board cells: {json.dumps(board)}\n"
        f"Your mark: {state.current_player}\n"
        f"Legal move indices: {json.dumps(state.get_legal_moves())}\n"
    )
