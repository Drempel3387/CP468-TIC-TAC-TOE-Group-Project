"""Terminal interface for Human-vs-AI and AI-vs-AI Tic-Tac-Toe."""
from __future__ import annotations

from game.board import Board
from game.stochastic import resolve_stochastic_move
from agents.human import HumanAgent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.alphabeta_agent import AlphaBetaAgent
from agents.expectiminimax_agent import ExpectiminimaxAgent
from agents.llm_agent import LLMAgent


def build_agents():
    agents = {
        "Random": RandomAgent(),
        "Minimax": MinimaxAgent(),
        "Alpha-Beta": AlphaBetaAgent(),
        "Expectiminimax": ExpectiminimaxAgent(),
    }
    gemini = LLMAgent()
    if gemini.is_available:
        agents["Gemini"] = gemini
    return agents


def display_board(state: Board):
    print("\n" + str(state) + "\n")


def _ask_choice(prompt, options):
    options = list(options)
    while True:
        print(prompt)
        for index, option in enumerate(options, 1):
            print(f"  {index}. {option}")
        raw = input("Select: ").strip()
        try:
            selected = int(raw) - 1
            if 0 <= selected < len(options):
                return options[selected]
        except ValueError:
            pass
        print(f"Enter a number from 1 to {len(options)}.")


def _ask_board_size():
    while True:
        raw = input("Board size N for an N x N board (3 recommended): ").strip()
        try:
            size = int(raw)
            if size >= 3:
                return size
        except ValueError:
            pass
        print("Please enter an integer of at least 3.")


def _ask_yes_no(prompt, default=False):
    suffix = "Y/n" if default else "y/N"
    answer = input(f"{prompt} ({suffix}): ").strip().lower()
    return default if not answer else answer in ("y", "yes")


def _select_agent(agents, prompt):
    name = _ask_choice(prompt, agents.keys())
    return agents[name]


def run_match(agent_x, agent_o, size=3, stochastic=False):
    """Run one match. Returns the final immutable Board state."""
    if not stochastic and any(isinstance(agent, ExpectiminimaxAgent)
                              for agent in (agent_x, agent_o)):
        print("Expectiminimax models random move outcomes, so stochastic mode is enabled.")
        stochastic = True
    state = Board.empty(size)
    agents = {"X": agent_x, "O": agent_o}
    display_board(state)
    while not state.is_terminal():
        player = state.current_player
        agent = agents[player]
        print(f"{player} ({getattr(agent, 'name', agent.__class__.__name__)}) to move.")
        move = agent.choose_move(state)
        if move not in state.get_legal_moves():
            raise ValueError(f"{getattr(agent, 'name', 'Agent')} selected illegal move {move!r}.")
        actual_move = resolve_stochastic_move(state, move) if stochastic else move
        if actual_move != move:
            print(f"Random event! Intended square {move + 1}, actual square {actual_move + 1}.")
        state = state.apply_move(actual_move)
        display_board(state)
    winner = state.winner()
    print(f"Player {winner} wins!" if winner else "The game is a draw!")
    for label, agent in (("X", agent_x), ("O", agent_o)):
        stats = getattr(agent, "last_stats", None)
        if stats is not None:
            print(f"{label} {getattr(agent, 'name', 'AI')} stats: "
                  f"{stats.nodes_evaluated} nodes, {stats.elapsed:.6f}s")
    return state


def human_vs_ai(agents):
    ai = _select_agent(agents, "Choose your AI opponent:")
    human_mark = _ask_choice("Choose your mark:", ("X", "O"))
    human = HumanAgent()
    if human_mark == "X":
        run_match(human, ai, size=_ask_board_size(), stochastic=_ask_yes_no("Enable stochastic moves?"))
    else:
        run_match(ai, human, size=_ask_board_size(), stochastic=_ask_yes_no("Enable stochastic moves?"))


def ai_vs_ai(agents):
    agent_x = _select_agent(agents, "Choose the AI playing as X:")
    agent_o = _select_agent(agents, "Choose the AI playing as O:")
    run_match(agent_x, agent_o, size=_ask_board_size(),
              stochastic=_ask_yes_no("Enable stochastic moves?"))


def main():
    agents = build_agents()
    while True:
        print("\n========== TIC-TAC-TOE ==========")
        print("1. Human vs AI")
        print("2. AI vs AI")
        print("3. How to play")
        print("4. List available agents")
        print("0. Exit")
        choice = input("Select an option: ").strip()
        try:
            if choice == "1":
                human_vs_ai(agents)
            elif choice == "2":
                ai_vs_ai(agents)
            elif choice == "3":
                print("\nChoose a square by entering its number. X and O take turns. "
                      "Three marks in a row, column, or diagonal wins. "
                      "A full board with no winner is a draw.")
                print("The default game is 3x3; larger N x N boards are also supported.")
                print("In stochastic mode, a move lands on its intended square 90% of the time.")
            elif choice == "4":
                print("Available agents: " + ", ".join(agents))
                if "Gemini" not in agents:
                    print("Gemini unavailable: set GEMINI_API_KEY to enable it.")
            elif choice == "0":
                print("Thanks for playing!")
                return
            else:
                print("Invalid menu option.")
        except (ValueError, RuntimeError) as exc:
            print(f"Could not complete that game: {exc}")
