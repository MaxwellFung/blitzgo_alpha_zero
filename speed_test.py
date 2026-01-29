#!/usr/bin/env python3
import time
import random
import numpy as np
from tqdm import tqdm
import az_engine  # The compiled C++ pybind module

# ================= CONFIGURATION =================
BOARD_SIZE = 13       # Set this to match your training size
N_GAMES = 10000      # Number of games to simulate
SEED = 42
# =================================================

def run_random_game(size: int):
    """
    Runs a single game making purely random moves until the game ends.
    Returns the number of moves made.
    """
    game = az_engine.Game(size)
    moves_count = 0
    max_moves = size * size * 4  # Safety cap

    while not game.is_over() and moves_count < max_moves:
        legal_moves = game.legal_moves_all()
        if not legal_moves:
            break

        move = random.choice(legal_moves)
        game.apply(move)
        moves_count += 1

    return moves_count

def benchmark():
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Starting benchmark: {N_GAMES} games on {BOARD_SIZE}x{BOARD_SIZE} board...")
    
    start_time = time.time()
    total_moves = 0

    for _ in tqdm(range(N_GAMES)):
        total_moves += run_random_game(BOARD_SIZE)

    end_time = time.time()
    duration = end_time - start_time

    # Calculations
    avg_time_per_game = duration / N_GAMES
    avg_time_per_move = duration / total_moves if total_moves > 0 else 0

    print("\n" + "="*40)
    print(f"BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Duration:     {duration:.4f} sec")
    print(f"Total Games:        {N_GAMES}")
    print(f"Total Moves:        {total_moves}")
    print("-" * 40)
    print(f"Avg Moves/Game:     {total_moves / N_GAMES:.2f}")
    print("-" * 40)
    print(f"Speed (Games/s):    {N_GAMES / duration:.2f}")
    print(f"Speed (Moves/s):    {total_moves / duration:.2f}")
    print("-" * 40)
    # The requested metrics:
    print(f"Avg Time per Game:  {avg_time_per_game * 1000:.4f} ms")
    print(f"Avg Time per Move:  {avg_time_per_move * 1_000_000:.4f} µs")
    print("="*40)

if __name__ == "__main__":
    benchmark()