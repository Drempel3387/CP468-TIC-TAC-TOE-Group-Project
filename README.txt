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

tests/                        Unit tests
  test_game.py                Game environment tests
  test_algorithms.py          Algorithm correctness tests
  test_agents.py              Agent behaviour tests
  test_gemini.py              Gemini integration tests

results/                      Benchmark output
  data/                       Raw result data
  figures/                    Generated charts

docs/                         Documentation
  report.md                   Source for the PDF design document
  poster/                     Presentation poster files
