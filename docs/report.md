# *CP468 – Multi-Agent Decision Making in Games | Final report | Standard 3×3 Tic-Tac-Toe*
*Group 1 - Adom, Michelle; Couto, Marie; Glennie, Robert; Mongeon, Ethan; Rempel, Devon* 



### Minimax

#### Overview

Minimax is an adversarial search algorithm for deterministic, two-player games. We used it to choose moves in standard 3×3 Tic-Tac-Toe. The algorithm assumes both players make optimal decisions: the AI is the maximizing player (**MAX**), and its opponent is the minimizing player (**MIN**).

The algorithm evaluates completed games using utility values from the AI's perspective:

| Game result | Utility |
|---|---:|
| AI wins | +1 |
| Draw | 0 |
| AI loses | −1 |

MAX chooses the highest possible value, while MIN chooses the lowest. By recursively considering the opponent's responses, Minimax identifies the move with the best outcome.

#### Implementation

Our Python implementation uses two functions in `algorithms/minimax.py`:

- `minimax(state, player, is_max, stats)` recursively evaluates possible future moves and alternates between MAX and MIN.
- `minimax_search(state, player)` evaluates each legal move and returns the best move, its utility score, and search statistics.

The `MinimaxAgent` class in `agents/minimax_agent.py` uses `choose_move(state)` so that the main game can request a move from the algorithm.

Instead of implementing Tic-Tac-Toe rules a second time, Minimax uses the game's shared state functions: `is_terminal()` to detect game endings, `get_legal_moves()` to list available moves, `apply_move()` to create the next state, and `utility(player)` to score a terminal state. The project’s `SearchStats` class records the number of visited nodes, terminal evaluations, and elapsed search time.

#### Pseudocode

```text
MINIMAX(state, player, is_max):
    if state is terminal:
        return utility(state, player)

    best = -infinity if is_max else +infinity

    for each legal move:
        next_state = apply_move(state, move)
        score = MINIMAX(next_state, player, not is_max)

        if is_max:
            best = max(best, score)
        else:
            best = min(best, score)

    return best

MINIMAX_SEARCH(state, player):
    best_move = None
    best_score = -infinity

    for each legal move:
        next_state = apply_move(state, move)
        score = MINIMAX(next_state, player, False)

        if score > best_score:
            best_score = score
            best_move = move

    return best_move, best_score
```

#### Complexity

The conventional worst-case time complexity of Minimax is **O(n^m)**, where **n** is the branching factor (available moves) and **m** is the maximum search depth. Standard 3×3 Tic-Tac-Toe has at most nine moves from an empty board, with fewer available after each turn. Our implementation explores all legal continuations without pruning, so early-game searches are more expensive than late-game searches.

The recursion uses **O(n)** active stack frames for a fixed-size board. Because `apply_move()` also creates a new board state at each recursive step, memory use depends on the size of each state.

#### Correctness Testing

We tested Minimax using Python's `unittest` framework. All four tests passed:

| Test | What it checks | Result |
|---|---|---|
| Empty board | The optimal game value is a draw (0) | Passed |
| Winning move | The AI selects an immediate winning move | Passed |
| Blocking move | The AI blocks an immediate opponent win | Passed |
| Agent interface | `MinimaxAgent` returns a legal move | Passed |

The four-test suite completed in **1.634 seconds**. This is the duration of the whole test suite, not the time taken for one Minimax move. The tests support correctness for these selected positions; integration with the completed game must also be checked.

Minimax also serves as the full-search baseline for the project's **Minimax versus Alpha-Beta** comparison. 

### Minimax and Alpha-Beta Performance Comparison

Both algorithms were evaluated on the same three standard 3×3 Tic-Tac-Toe states. For each state and algorithm, the search was run three times and the median execution time reported. Both algorithms used full-depth search without heuristics.

| Board state | Algorithm | Nodes evaluated | Median time (ms) | Optimal utility |
|---|---|---:|---:|---:|
| Empty | Minimax | 549,946 | 1,019.710 | 0 |
| Empty | Alpha-Beta | 18,297 | 40.197 | 0 |
| Middle | Minimax | 182 | 0.352 | +1 |
| Middle | Alpha-Beta | 97 | 0.220 | +1 |
| Late | Minimax | 8 | 0.016 | +1 |
| Late | Alpha-Beta | 8 | 0.017 | +1 |

On the empty board, Alpha-Beta evaluated **96.7% fewer nodes** (531,649 fewer) and took approximately **25.4 times less time** than Minimax. On the middle-game state, it evaluated **46.7% fewer nodes**. Both algorithms returned the same optimal utility for every tested state. On the late-game state, both evaluated eight nodes and had nearly identical runtimes. Since the timing results are medians from only three runs, small differences at sub-millisecond scales should not be interpreted as meaningful.

