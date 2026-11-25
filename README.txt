SETUP
=====

Requires Python 3.9 or newer (developed on 3.13).

1. Install the dependency (only needed for plots; the game itself uses the
   standard library only):
       pip install -r requirements.txt

2. (Optional) Enable the Gemini agent. Get a free API key from Google AI Studio
   and set it in your shell (the program does not read .env automatically):
       export GEMINI_API_KEY="your-key"
       export GEMINI_MODEL="gemini-2.0-flash"      # optional
   Without a key the game still works; Gemini simply isn't listed as an agent.
   Gemini settings are documented in config/gemini_config.json.


RUNNING
=======

Play (run from the project root):
    python3 main.py
  Menu: 1 = Human vs AI, 2 = AI vs AI. You can pick the agent (Random, Minimax,
  Alpha-Beta, Expectiminimax, Gemini), the board size (3x3 default, larger N x N
  supported) and whether to enable stochastic moves (a move lands on the
  intended square 90% of the time; use Expectiminimax for this mode).

Tests:
    python3 -m unittest discover -s tests -t .

Benchmarks (results are written to results/data, run from the project root):
    python3 -m evaluation.benchmark                 # standard + stochastic 3x3 games
    python3 -m evaluation.benchmark --probe         # time one move per algorithm
    python3 -m evaluation.scalability               # 3x3 / 4x4 / 5x5 cost and play
    python3 -m evaluation.gemini_benchmark --thinking low      # needs GEMINI_API_KEY
    python3 -m evaluation.gemini_benchmark --thinking minimal --out results/data/gemini_games_minimal.csv
    python3 -m evaluation.plots                     # redraw figures in results/figures

Poster trees:
    python3 docs/poster/make_trees.py

Note: full-depth Minimax on an empty 3x3 board evaluates about 550,000 nodes
(a few seconds). Use Alpha-Beta or a depth limit for larger boards.


PROJECT STRUCTURE
=================

main.py                       Entry point: select mode (Human vs AI / AI vs AI), agents, board size
README.txt                    This file
requirements.txt              Python dependencies
.env.example                  Template for environment variables (Gemini API key)
.gitignore                    Git ignore rules

game/                         Game environment
  board.py                    N x N board representation, move validation, win/draw detection
  tictactoe.py                Tic-Tac-Toe rules and game loop
  stochastic.py               Stochastic variant with chance elements (for Expectiminimax)

algorithms/                   Search algorithms
  minimax.py                  Minimax
  alphabeta.py                Alpha-Beta Pruning
  expectiminimax.py           Expectiminimax (max, min and chance nodes)
  heuristics.py               Heuristic evaluation for depth-limited search on larger boards
  stats.py                    Node count and execution time tracking

agents/                       Players (common interface in base.py)
  base.py                     Abstract agent interface
  human.py                    Human player
  random_agent.py             Random baseline player
  minimax_agent.py            Minimax agent
  alphabeta_agent.py          Alpha-Beta agent
  expectiminimax_agent.py     Expectiminimax agent
  llm_agent.py                Gemini-based agent

api/                          Gemini API integration
  gemini_client.py            API client wrapper
  prompts.py                  Prompt templates for move generation

config/                       Configuration files
  gemini_config.json          Gemini model and agent settings
  settings.json               General game and benchmark settings

ui/                           Visualization
  cli.py                      Text-based interface for game states and decisions
  tree_viz.py                 Game tree display with pruned branches highlighted

evaluation/                   Testing and performance comparison
  benchmark.py                Runs matches across algorithms and board sizes
  metrics.py                  Execution time, node evaluations, win rates
  plots.py                    Charts and tables for the report and poster
  scalability.py              Search cost and play strength on 3x3, 4x4, 5x5 boards
  gemini_benchmark.py         Plays Gemini against the search agents (needs API key)

tests/                        Unit tests
  test_game.py                Game environment tests
  test_algorithms.py          Algorithm correctness tests
  test_agents.py              Agent behaviour tests
  test_expectiminimax.py      Expectiminimax and chance-move tests
  test_gemini.py              Gemini integration tests

results/                      Benchmark output
  data/                       Raw result data
  figures/                    Generated charts

docs/                         Documentation
  report.md                   Source for the PDF design document
  poster/                     Presentation poster (poster.html / poster.pdf)
    make_trees.py             Generates the game-tree images used on the poster
