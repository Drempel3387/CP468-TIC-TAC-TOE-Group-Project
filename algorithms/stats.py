"""Node count and execution time tracking for search algorithms."""

import time
from dataclasses import dataclass


@dataclass
class SearchStats:
    nodes_evaluated: int = 0   # every state visited by the search
    terminal_evals: int = 0    # leaves scored (terminal or depth cutoff)
    prunes: int = 0            # number of alpha/beta cutoffs triggered
    pruned_branches: int = 0   # total child moves skipped by those cutoffs
    elapsed: float = 0.0       # seconds
    _start: float = 0.0

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        self.elapsed = time.perf_counter() - self._start
