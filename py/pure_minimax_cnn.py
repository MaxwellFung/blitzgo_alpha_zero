#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import az_engine


BOARD_SIZE = 7
VALUE_SCALE = 500.0
INF = 1_000_000_000


def encode_for_player(game: az_engine.Game, player: int) -> np.ndarray:
    size = game.size()
    opponent = 2 if player == 1 else 1
    stones = np.asarray(game.stones(), dtype=np.uint8).reshape(size, size)
    territories = np.asarray(game.territories(), dtype=np.uint8).reshape(size, size)
    return np.stack(
        (
            stones == player,
            stones == opponent,
            territories == player,
            territories == opponent,
            stones == 0,
        ),
        dtype=np.uint8,
    )


def terminal_value(game: az_engine.Game, root_player: int) -> int:
    if root_player == 1:
        diff = game.score_p1() - game.score_p2()
    else:
        diff = game.score_p2() - game.score_p1()
    return int(diff * 100)


class CnnLeafMinimax:
    def __init__(self, model_path: str | Path, depth: int, torch_threads: int = 1):
        torch.set_num_threads(max(1, torch_threads))
        self.model = torch.jit.load(str(model_path), map_location="cpu").eval()
        self.depth = max(1, depth)
        self.states = 0

    def evaluate_leaf(self, game: az_engine.Game, root_player: int) -> int:
        board = torch.from_numpy(encode_for_player(game, root_player)).unsqueeze(0)
        with torch.no_grad():
            value = float(self.model(board).squeeze().item())
        if not math.isfinite(value):
            return 0
        return int(round(max(-1.0, min(1.0, value)) * VALUE_SCALE))

    def legal_moves(self, game: az_engine.Game) -> list[int]:
        return list(game.legal_moves_playable())

    def search(self, game: az_engine.Game, depth: int, alpha: int, beta: int, root_player: int) -> int:
        self.states += 1
        if game.is_over():
            return terminal_value(game, root_player)
        if depth == 0:
            return self.evaluate_leaf(game, root_player)

        moves = self.legal_moves(game)
        if not moves:
            return self.evaluate_leaf(game, root_player)

        maximizing = game.current_player() == root_player
        if maximizing:
            best = -INF
            for move in moves:
                if game.apply(move) != 0:
                    continue
                best = max(best, self.search(game, depth - 1, alpha, beta, root_player))
                game.undo()
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best

        best = INF
        for move in moves:
            if game.apply(move) != 0:
                continue
            best = min(best, self.search(game, depth - 1, alpha, beta, root_player))
            game.undo()
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    def best_move(self, game: az_engine.Game) -> tuple[int, int]:
        self.states = 0
        root_player = game.current_player()
        best_move = -1
        best_value = -INF
        alpha = -INF
        for move in self.legal_moves(game):
            if game.apply(move) != 0:
                continue
            value = self.search(game, self.depth - 1, alpha, INF, root_player)
            game.undo()
            if best_move < 0 or value > best_value:
                best_move = move
                best_value = value
            alpha = max(alpha, best_value)
        return best_move, best_value


def print_board(game: az_engine.Game):
    stones = game.stones()
    territories = game.territories()
    print("\nBoard:")
    for y in range(game.size()):
        row = []
        for x in range(game.size()):
            pos = y * game.size() + x
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


def ask_human_move(game: az_engine.Game) -> int:
    while True:
        raw = input("Your move (x y) [1-based]: ").strip()
        try:
            x1, y1 = map(int, raw.split())
            x, y = x1 - 1, y1 - 1
            if not (0 <= x < game.size() and 0 <= y < game.size()):
                print("Out of bounds.")
                continue
            move = y * game.size() + x
            if game.apply(move) == 0:
                return move
            print("Illegal move.")
        except ValueError:
            print("Enter: x y")


def main():
    parser = argparse.ArgumentParser(description="Pure minimax with a CNN leaf evaluator.")
    parser.add_argument("--model", type=Path, default=Path("model/minimax_value_cnn.ts"))
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Missing {args.model}. Train it with "
            "`python3 training/train_minimax_value_cnn.py`."
        )

    game = az_engine.Game(BOARD_SIZE)
    engine = CnnLeafMinimax(args.model, args.depth, args.torch_threads)
    side = input("Play as player 1 (X) or player 2 (O)? [1/2]: ").strip()
    human_side = 1 if side != "2" else 2
    print(f"Pure minimax depth={args.depth}, CNN leaf model={args.model}")

    while not game.is_over():
        print_board(game)
        if game.current_player() == human_side:
            ask_human_move(game)
            continue

        move, value = engine.best_move(game)
        if move < 0:
            raise RuntimeError("No legal engine move found.")
        game.apply(move)
        print(
            f"Engine plays: ({move % BOARD_SIZE + 1}, {move // BOARD_SIZE + 1}) "
            f"value={value} searched={engine.states:,}"
        )

    print_board(game)
    winner = game.winner()
    print("Draw." if winner == 0 else f"Player {winner} wins.")


if __name__ == "__main__":
    main()
