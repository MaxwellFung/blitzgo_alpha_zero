#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import queue
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np
import torch
from tqdm import tqdm

import az_engine
from move_ranker import ranked_moves

torch.set_num_threads(1)


BOARD_SIZE = 7
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
ILLEGAL_SCORE = np.iinfo(np.int16).min
PROGRESS_QUEUE = None


def initialize_worker(progress_queue):
    global PROGRESS_QUEUE
    PROGRESS_QUEUE = progress_queue


def encode_game(game) -> np.ndarray:
    stones = np.asarray(game.stones(), dtype=np.uint8).reshape(BOARD_SIZE, BOARD_SIZE)
    territories = np.asarray(game.territories(), dtype=np.uint8).reshape(BOARD_SIZE, BOARD_SIZE)
    player = game.current_player()
    opponent = 2 if player == 1 else 1
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


def choose_played_move(
    rng: random.Random,
    moves: list[int],
    scores: list[int],
    best_probability: float,
    top_k_probability: float,
    top_k: int,
) -> int:
    ranked = sorted(zip(moves, scores), key=lambda item: (-item[1], item[0]))
    roll = rng.random()
    if roll < best_probability:
        return ranked[0][0]
    if roll < best_probability + top_k_probability:
        return rng.choice(ranked[: min(top_k, len(ranked))])[0]
    return rng.choice(ranked)[0]


def candidate_moves(model, game, top_k: int, extra_random: int, rng: random.Random) -> list[int]:
    legal = list(game.legal_moves_playable())
    if not legal:
        return []
    candidates = ranked_moves(model, game)[: min(top_k, len(legal))]
    if extra_random <= 0:
        return candidates

    seen = set(candidates)
    extras = [move for move in legal if move not in seen]
    rng.shuffle(extras)
    candidates.extend(extras[:extra_random])
    return candidates


def generate_shard(task: tuple) -> dict:
    (
        worker_id,
        shard_id,
        sample_count,
        teacher_states,
        seed,
        output_dir,
        max_game_moves,
        progress_every,
        best_probability,
        top_k_probability,
        exploration_top_k,
        sample_probability,
        max_samples_per_game,
        search_workers,
        ranker_path,
        root_top_k,
        internal_top_k,
        extra_random_moves,
    ) = task
    started = time.monotonic()
    rng = random.Random(seed)
    model = torch.jit.load(ranker_path, map_location="cpu")
    model.eval()
    boards = np.empty((sample_count, 5, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    legal_masks = np.zeros((sample_count, ACTION_SIZE), dtype=np.bool_)
    move_scores = np.full((sample_count, ACTION_SIZE), ILLEGAL_SCORE, dtype=np.int16)
    completed_depths = np.empty(sample_count, dtype=np.int16)

    sample_index = 0
    pending_progress = 0
    games_played = 0
    while sample_index < sample_count:
        games_played += 1
        game_samples = 0
        game = az_engine.Game(BOARD_SIZE)

        for _ in range(max_game_moves):
            if game.is_over() or sample_index >= sample_count:
                break

            candidates = candidate_moves(model, game, root_top_k, extra_random_moves, rng)
            if not candidates:
                break
            teacher = az_engine.Minimax(
                max_states=teacher_states,
                internal_top_k=internal_top_k,
            )
            labels = teacher.best_move_subset_parallel_info(
                game,
                candidates,
                search_workers,
            )
            moves = list(labels["moves"])
            scores = list(labels["scores"])
            if not moves:
                break

            should_sample = (
                game_samples < max_samples_per_game
                and rng.random() < sample_probability
            )
            if should_sample:
                boards[sample_index] = encode_game(game)
                legal_masks[sample_index, moves] = True
                move_scores[sample_index, moves] = np.asarray(scores, dtype=np.int16)
                completed_depths[sample_index] = labels["completed_depth"]
                sample_index += 1
                game_samples += 1
                pending_progress += 1
                if pending_progress >= progress_every:
                    PROGRESS_QUEUE.put(pending_progress)
                    pending_progress = 0

            move = choose_played_move(
                rng,
                moves,
                scores,
                best_probability,
                top_k_probability,
                exploration_top_k,
            )
            if game.apply(move) != 0:
                raise RuntimeError(f"Teacher selected illegal move {move}")

    if pending_progress:
        PROGRESS_QUEUE.put(pending_progress)

    path = Path(output_dir) / f"shard_{shard_id:03d}.npz"
    np.savez_compressed(
        path,
        boards=boards,
        legal_masks=legal_masks,
        move_scores=move_scores,
        completed_depths=completed_depths,
    )
    return {
        "worker_id": worker_id,
        "shard_id": shard_id,
        "samples": sample_count,
        "games": games_played,
        "average_depth": float(completed_depths.mean()),
        "seconds": time.monotonic() - started,
        "path": str(path),
    }


def split_samples(total: int, workers: int) -> list[int]:
    base, extra = divmod(total, workers)
    return [base + (worker < extra) for worker in range(workers)]


def next_shard_id(output_dir: Path) -> int:
    highest = -1
    for path in output_dir.glob("shard_*.npz"):
        try:
            highest = max(highest, int(path.stem.removeprefix("shard_")))
        except ValueError:
            continue
    return highest + 1


def main():
    parser = argparse.ArgumentParser(description="Generate CNN move-ranking teacher data.")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--teacher-states", type=int, default=100_000)
    parser.add_argument("--ranker", default="model/move_ranker.ts")
    parser.add_argument("--root-top-k", type=int, default=12)
    parser.add_argument("--internal-top-k", type=int, default=0)
    parser.add_argument("--extra-random-moves", type=int, default=2)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--search-workers",
        type=int,
        default=1,
        help=(
            "CPU threads per minimax root search. Use 10 with --workers 1 to "
            "parallelize each teacher search across a 10-core machine."
        ),
    )
    parser.add_argument("--output-dir", default="data/move_ranker_7x7")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-game-moves", type=int, default=BOARD_SIZE * BOARD_SIZE * 4)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--best-probability", type=float, default=0.65)
    parser.add_argument("--top-k-probability", type=float, default=0.25)
    parser.add_argument("--exploration-top-k", type=int, default=5)
    parser.add_argument("--sample-probability", type=float, default=0.50)
    parser.add_argument("--max-samples-per-game", type=int, default=32)
    args = parser.parse_args()

    workers = max(1, min(args.workers, args.samples))
    search_workers = max(1, args.search_workers)
    progress_every = max(1, args.progress_every)
    if not 0.0 <= args.best_probability <= 1.0:
        raise ValueError("--best-probability must be between 0 and 1.")
    if not 0.0 <= args.top_k_probability <= 1.0:
        raise ValueError("--top-k-probability must be between 0 and 1.")
    if args.best_probability + args.top_k_probability > 1.0:
        raise ValueError("--best-probability + --top-k-probability must be <= 1.")
    if not 0.0 < args.sample_probability <= 1.0:
        raise ValueError("--sample-probability must be in (0, 1].")
    if args.exploration_top_k < 1:
        raise ValueError("--exploration-top-k must be at least 1.")
    if args.max_samples_per_game < 1:
        raise ValueError("--max-samples-per-game must be at least 1.")
    if args.root_top_k < 1:
        raise ValueError("--root-top-k must be at least 1.")
    if args.internal_top_k < 0:
        raise ValueError("--internal-top-k must be at least 0.")
    if args.extra_random_moves < 0:
        raise ValueError("--extra-random-moves must be at least 0.")
    ranker_path = Path(args.ranker)
    if not ranker_path.exists():
        raise FileNotFoundError(f"Missing TorchScript move ranker: {ranker_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_shard_id = next_shard_id(output_dir)

    print(
        f"Generating {args.samples:,} positions with {workers} CPU workers "
        f"and {args.teacher_states:,} teacher states per position."
    )
    print(f"Minimax root search workers per position: {search_workers}.")
    print(
        f"CNN-guided minimax: ranker={ranker_path}, root_top_k={args.root_top_k}, "
        f"extra_random_moves={args.extra_random_moves}, internal_top_k={args.internal_top_k}."
    )
    if workers > 1 and search_workers > 1:
        print(
            "Warning: --workers multiplied by --search-workers can oversubscribe "
            "your CPU. For 10 total cores, prefer --workers 1 --search-workers 10 "
            "or another product near 10."
        )
    print(
        "Self-play policy: "
        f"{args.best_probability:.0%} best, "
        f"{args.top_k_probability:.0%} top-{args.exploration_top_k}, "
        f"{1.0 - args.best_probability - args.top_k_probability:.0%} random. "
        f"Sampling {args.sample_probability:.0%} of visited positions, "
        f"max {args.max_samples_per_game} samples/game."
    )
    if first_shard_id:
        print(f"Appending new shards starting at shard_{first_shard_id:03d}.npz.")
    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    tasks = [
        (
            worker_id,
            first_shard_id + worker_id,
            sample_count,
            args.teacher_states,
            args.seed + first_shard_id + worker_id,
            str(output_dir),
            args.max_game_moves,
            progress_every,
            args.best_probability,
            args.top_k_probability,
            args.exploration_top_k,
            args.sample_probability,
            args.max_samples_per_game,
            search_workers,
            str(ranker_path),
            args.root_top_k,
            args.internal_top_k,
            args.extra_random_moves,
        )
        for worker_id, sample_count in enumerate(split_samples(args.samples, workers))
    ]

    with context.Pool(
        workers,
        initializer=initialize_worker,
        initargs=(progress_queue,),
    ) as pool:
        results = [pool.apply_async(generate_shard, (task,)) for task in tasks]
        pending = list(results)
        with tqdm(total=args.samples, unit="position", dynamic_ncols=True) as progress:
            while pending:
                try:
                    progress.update(progress_queue.get(timeout=0.2))
                except queue.Empty:
                    pass

                for result in list(pending):
                    if not result.ready():
                        continue
                    summary = result.get()
                    pending.remove(result)
                    tqdm.write(
                        f"worker={summary['worker_id']:02d} "
                        f"shard={summary['shard_id']:03d} "
                        f"samples={summary['samples']:,} "
                        f"games={summary['games']:,} "
                        f"avg_depth={summary['average_depth']:.2f} "
                        f"elapsed={summary['seconds']:.1f}s "
                        f"wrote={summary['path']}"
                    )

            while True:
                try:
                    progress.update(progress_queue.get_nowait())
                except queue.Empty:
                    break


if __name__ == "__main__":
    main()
