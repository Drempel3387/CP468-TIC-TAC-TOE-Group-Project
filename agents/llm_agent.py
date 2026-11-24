"""Gemini-backed move-selection agent."""
import json
import re
from .base import Agent
from api.gemini_client import GeminiClient
from api.prompts import move_prompt


class LLMAgent(Agent):
    name = "Gemini"

    def __init__(self, client=None):
        self.client = client or GeminiClient()

    @property
    def is_available(self):
        return self.client.is_configured

    def choose_move(self, state):
        text = self.client.generate(move_prompt(state))
        try:
            move = json.loads(text)["move"]
        except (json.JSONDecodeError, KeyError, TypeError):
            match = re.search(r"\b\d+\b", text)
            if not match:
                raise RuntimeError("Gemini response did not contain a move index.")
            move = int(match.group())
        if not isinstance(move, int) or isinstance(move, bool) or move not in state.get_legal_moves():
            raise RuntimeError(f"Gemini returned an illegal move: {move!r}")
        return move
