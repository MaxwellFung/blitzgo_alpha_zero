#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import az_engine

# ================== CONFIG ==================

BOARD_SIZE = 7
NUM_STATES_SEARCHED = 10000000
RANKER_MODEL = "model/move_ranker_7x7.ts"
TOP_K = 12
WORKERS = os.cpu_count() or 1

# ============================================


def print_board(game: az_engine.Game, size: int):
    stones = game.stones()
    territories = game.territories()

    print("\nBoard:")
    for y in range(size):
        row = []
        for x in range(size):
            pos = y * size + x
            if stones[pos] == 1:
                row.append("X")
            elif stones[pos] == 2:
                row.append("O")
            elif territories[pos] == 1:
                row.append("x")
            elif territories[pos] == 2:
                row.append("o")
            else:
                row.append(".")
        print(" ".join(row))
    print(f"Territory: X={game.score_p1()} O={game.score_p2()}\n")


def ask_human_move(game: az_engine.Game, size: int) -> int:
    while True:
        value = input("Your move (x y) [1-based]: ").strip()
        try:
            x1, y1 = map(int, value.split())
            x = x1 - 1
            y = y1 - 1
            if not (0 <= x < size and 0 <= y < size):
                print("Out of bounds.")
                continue
            move = y * size + x
            if game.apply(move) == 0:
                return move
            print("Illegal move.")
        except ValueError:
            print("Enter: x y   (both from 1 to N)")


def print_winner(game: az_engine.Game):
    winner = game.winner()
    if winner == 0:
        print("Draw.")
    else:
        print(f"Player {winner} wins!")


def play_pvp(game: az_engine.Game, size: int):
    print("\nPlayer vs Player mode.")
    print("Player 1 = X, Player 2 = O\n")

    while not game.is_over():
        print_board(game, size)
        print(f"Player {game.current_player()} to move.")
        ask_human_move(game, size)

    print_board(game, size)
    print_winner(game)


def play_minimax(
    game: az_engine.Game,
    size: int,
    max_states: int,
    ranker_model: str,
    top_k: int,
    workers: int,
):
    from move_ranker import load_scripted_model, ranked_moves

    if not Path(ranker_model).exists():
        raise FileNotFoundError(
            f"Missing 7x7 ranker model: {ranker_model}. "
            "Generate data with `python3 py/generate_training_data.py`, then train with "
            "`python3 py/train_move_ranker.py`."
        )

    search = az_engine.Minimax(max_states=max_states)
    model = load_scripted_model(ranker_model)

    player_name = f"CNN top-{top_k} parallel minimax"
    print(
        f"\nPlay against {player_name} with {max_states:,} searched states per move "
        f"using {ranker_model} and {workers} CPU workers."
    )
    side = input("Play as player 1 (X) or player 2 (O)? [1/2]: ").strip()
    human_side = 1 if side != "2" else 2

    while not game.is_over():
        print_board(game, size)
        if game.current_player() == human_side:
            ask_human_move(game, size)
            continue

        candidates = ranked_moves(model, game)[:top_k]
        move = search.best_move_subset_parallel(game, candidates, workers)
        if move < 0:
            raise RuntimeError("Top-k minimax found no legal move.")
        game.apply(move)
        print(
            f"{player_name} plays: ({move % size + 1}, {move // size + 1}) "
            f"[kept {len(candidates)} root moves, "
            f"searched {search.states_searched():,} states, "
            f"completed depth {search.completed_depth()}]"
        )

    print_board(game, size)
    print_winner(game)


def main():
    parser = argparse.ArgumentParser(description="Play BlitzGo with CNN top-k minimax.")
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE)
    parser.add_argument("--states", type=int, default=NUM_STATES_SEARCHED)
    parser.add_argument(
        "--ranker",
        default=RANKER_MODEL,
        help="TorchScript move-ranker model.",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    if args.board_size != BOARD_SIZE:
        raise SystemExit(
            f"The CNN ranker currently supports only {BOARD_SIZE}x{BOARD_SIZE}; "
            f"got --board-size {args.board_size}."
        )

    print("Select mode:")
    print("  1) Human vs Minimax")
    print("  2) Human vs Human (PvP)")
    mode = input("Choose mode [1/2]: ").strip()

    game = az_engine.Game(args.board_size)
    if mode == "2":
        play_pvp(game, args.board_size)
    else:
        play_minimax(
            game,
            args.board_size,
            args.states,
            args.ranker,
            args.top_k,
            max(1, args.workers),
        )


if __name__ == "__main__":
    main()
