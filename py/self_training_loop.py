#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import shutil
import sys
import time
from pathlib import Path

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import az_engine
from move_ranker import ACTION_SIZE, BOARD_SIZE, CHANNELS, MoveRanker, encode_game, ranked_moves

torch.set_num_threads(1)


ILLEGAL_SCORE = np.iinfo(np.int16).min
PROGRESS_QUEUE = None


def cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


class LocalProgressQueue:
    def __init__(self, progress):
        self.progress = progress

    def put(self, count: int):
        self.progress.update(count)


def initialize_worker(progress_queue):
    global PROGRESS_QUEUE
    PROGRESS_QUEUE = progress_queue


def shard_count(data_dir: Path) -> int:
    return len(list(data_dir.glob("shard_*.npz")))


def sample_count(data_dir: Path) -> int:
    total = 0
    for path in data_dir.glob("shard_*.npz"):
        with np.load(path) as shard:
            total += int(shard["boards"].shape[0])
    return total


def next_shard_id(data_dir: Path) -> int:
    highest = -1
    for path in data_dir.glob("shard_*.npz"):
        try:
            highest = max(highest, int(path.stem.removeprefix("shard_")))
        except ValueError:
            continue
    return highest + 1


def split_samples(total: int, workers: int) -> list[int]:
    base, extra = divmod(total, workers)
    return [base + (worker < extra) for worker in range(workers)]


def save_random_model(output: Path, checkpoint: Path, channels: int, blocks: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = MoveRanker(channels=channels, blocks=blocks).eval()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": channels,
            "blocks": blocks,
            "board_size": BOARD_SIZE,
            "random_initialization": True,
        },
        checkpoint,
    )
    scripted = torch.jit.trace(model, torch.zeros(1, CHANNELS, BOARD_SIZE, BOARD_SIZE))
    scripted.save(str(output))


def load_shards(data_dir: Path):
    paths = sorted(data_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {data_dir}")
    shards = [np.load(path) for path in paths]
    boards = np.concatenate([shard["boards"] for shard in shards])
    legal_masks = np.concatenate([shard["legal_masks"] for shard in shards])
    move_scores = np.concatenate([shard["move_scores"] for shard in shards])
    return boards, legal_masks, move_scores


def ranking_loss(logits, legal_masks, move_scores, temperature: float):
    masked_logits = logits.masked_fill(~legal_masks, torch.finfo(logits.dtype).min)
    masked_scores = move_scores.float().masked_fill(
        ~legal_masks,
        torch.finfo(logits.dtype).min,
    )
    targets = F.softmax(masked_scores / temperature, dim=1)
    log_probs = F.log_softmax(masked_logits, dim=1)
    return F.kl_div(log_probs, targets, reduction="batchmean", log_target=False)


def topk_accuracy(logits, legal_masks, move_scores, k: int):
    masked_logits = logits.masked_fill(~legal_masks, -1e9)
    masked_scores = move_scores.masked_fill(~legal_masks, -32768)
    k = min(k, masked_logits.shape[1])
    best_teacher = masked_scores.argmax(dim=1, keepdim=True)
    topk_model = masked_logits.topk(k, dim=1).indices
    return (topk_model == best_teacher).any(dim=1).float().mean()


def transform_grid(grid: torch.Tensor, transform: int):
    if transform >= 4:
        grid = torch.flip(grid, dims=(-1,))
        transform -= 4
    if transform:
        grid = torch.rot90(grid, transform, dims=(-2, -1))
    return grid


def augment_batch(board, legal_masks, move_scores):
    transform = int(torch.randint(0, 8, ()).item())
    board = transform_grid(board, transform)
    legal_masks = transform_grid(
        legal_masks.view(-1, BOARD_SIZE, BOARD_SIZE),
        transform,
    ).reshape(-1, ACTION_SIZE)
    move_scores = transform_grid(
        move_scores.view(-1, BOARD_SIZE, BOARD_SIZE),
        transform,
    ).reshape(-1, ACTION_SIZE)
    return board, legal_masks, move_scores


def run_training(args, iteration: int, model_path: Path, checkpoint_path: Path):
    torch.manual_seed(args.seed + iteration)
    np.random.seed(args.seed + iteration)
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}; torch_threads={args.torch_threads}.")
    boards, legal_masks, move_scores = load_shards(args.data_dir)
    dataset = TensorDataset(
        torch.from_numpy(boards),
        torch.from_numpy(legal_masks),
        torch.from_numpy(move_scores),
    )
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed + iteration),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        num_workers=args.loader_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = MoveRanker(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_loss = float("inf")
    best_state = None

    print(f"Training on {train_size:,} positions; validating on {val_size:,}.")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for board, mask, scores in train_loader:
            board = board.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)
            if not args.no_augment:
                board, mask, scores = augment_batch(board, mask, scores)
            optimizer.zero_grad(set_to_none=True)
            loss = ranking_loss(model(board), mask, scores, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            train_loss += float(loss.detach()) * len(board)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_top1 = 0.0
        val_top3 = 0.0
        val_top5 = 0.0
        val_top10 = 0.0
        with torch.no_grad():
            for board, mask, scores in val_loader:
                board = board.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                scores = scores.to(device, non_blocking=True)
                logits = model(board)
                loss = ranking_loss(logits, mask, scores, args.temperature)
                val_loss += float(loss.detach()) * len(board)
                val_top1 += float(topk_accuracy(logits, mask, scores, 1)) * len(board)
                val_top3 += float(topk_accuracy(logits, mask, scores, 3)) * len(board)
                val_top5 += float(topk_accuracy(logits, mask, scores, 5)) * len(board)
                val_top10 += float(topk_accuracy(logits, mask, scores, 10)) * len(board)

        epoch_val_loss = val_loss / val_size
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss / train_size:.4f} "
            f"val_loss={epoch_val_loss:.4f} "
            f"val_top1={val_top1 / val_size:.3f} "
            f"val_top3={val_top3 / val_size:.3f} "
            f"val_top5={val_top5 / val_size:.3f} "
            f"val_top10={val_top10 / val_size:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.trace(
        model,
        torch.zeros(1, CHANNELS, BOARD_SIZE, BOARD_SIZE),
    )
    scripted.save(str(model_path))

    checkpoint_path.write_text(
        json.dumps(
            {
                "iteration": iteration,
                "model": str(model_path),
                "data_dir": str(args.data_dir),
                "samples": sample_count(args.data_dir),
                "best_val_loss": best_val_loss,
                "created_at": time.time(),
            },
            indent=2,
        )
        + "\n"
    )


def candidate_moves(model, game, top_k: int, extra_random: int, rng: random.Random) -> list[int]:
    legal = list(game.legal_moves_playable())
    if not legal:
        return []
    ranked = ranked_moves(model, game) if model is not None else legal
    candidates = []
    seen = set()
    for move in ranked[: min(top_k, len(ranked))]:
        if move not in seen:
            seen.add(move)
            candidates.append(move)
    if extra_random > 0:
        extras = [move for move in legal if move not in seen]
        rng.shuffle(extras)
        for move in extras[:extra_random]:
            seen.add(move)
            candidates.append(move)
    return candidates


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


def generate_guided_shard(task: tuple) -> dict:
    (
        worker_id,
        shard_id,
        sample_target,
        seed,
        output_dir,
        model_path,
        teacher_states,
        root_top_k,
        internal_top_k,
        search_workers,
        extra_random_moves,
        best_probability,
        top_k_probability,
        exploration_top_k,
        sample_probability,
        max_samples_per_game,
        max_game_moves,
        progress_every,
    ) = task
    started = time.monotonic()
    rng = random.Random(seed)
    model = None
    if model_path:
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()

    boards = np.empty((sample_target, CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    legal_masks = np.zeros((sample_target, ACTION_SIZE), dtype=np.bool_)
    move_scores = np.full((sample_target, ACTION_SIZE), ILLEGAL_SCORE, dtype=np.int16)
    completed_depths = np.empty(sample_target, dtype=np.int16)

    sample_index = 0
    pending_progress = 0
    games_played = 0
    while sample_index < sample_target:
        games_played += 1
        game_samples = 0
        game = az_engine.Game(BOARD_SIZE)

        for _ in range(max_game_moves):
            if game.is_over() or sample_index >= sample_target:
                break

            candidates = candidate_moves(model, game, root_top_k, extra_random_moves, rng)
            if not candidates:
                break

            teacher = az_engine.Minimax(
                max_states=teacher_states,
                internal_top_k=internal_top_k,
            )
            info = teacher.best_move_subset_parallel_info(
                game,
                candidates,
                search_workers,
            )
            moves = list(info["moves"])
            scores = list(info["scores"])
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
                completed_depths[sample_index] = int(info["completed_depth"])
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
                raise RuntimeError(f"Self-play selected illegal move {move}")

    if pending_progress:
        PROGRESS_QUEUE.put(pending_progress)

    path = Path(output_dir) / f"shard_{shard_id:03d}.npz"
    np.savez_compressed(
        path,
        boards=boards,
        legal_masks=legal_masks,
        move_scores=move_scores,
        completed_depths=completed_depths,
        iteration=np.asarray([shard_id], dtype=np.int32),
    )
    return {
        "worker_id": worker_id,
        "shard_id": shard_id,
        "samples": sample_target,
        "games": games_played,
        "average_depth": float(completed_depths.mean()),
        "seconds": time.monotonic() - started,
        "path": str(path),
    }


def generate_guided_data(args, iteration: int, model_path: Path):
    torch.set_num_threads(1)
    workers = max(
        1,
        min(args.workers or cpu_count() // max(1, args.search_workers), args.samples_per_iteration),
    )
    first_shard = next_shard_id(args.data_dir)
    if workers == 1:
        global PROGRESS_QUEUE
        task = (
            0,
            first_shard,
            args.samples_per_iteration,
            args.seed + iteration * 100_000,
            str(args.data_dir),
            str(model_path),
            args.teacher_states,
            args.root_top_k,
            args.internal_top_k,
            args.search_workers,
            args.extra_random_moves,
            args.best_probability,
            args.top_k_probability,
            args.exploration_top_k,
            args.sample_probability,
            args.max_samples_per_game,
            args.max_game_moves,
            args.progress_every,
        )
        print(
            f"Generating {args.samples_per_iteration:,} CNN-guided positions "
            f"with 1 local worker, root_top_k={args.root_top_k}, "
            f"internal_top_k={args.internal_top_k}."
        )
        with tqdm(total=args.samples_per_iteration, unit="position", dynamic_ncols=True) as progress:
            PROGRESS_QUEUE = LocalProgressQueue(progress)
            summary = generate_guided_shard(task)
            PROGRESS_QUEUE = None
            tqdm.write(
                f"worker={summary['worker_id']:02d} "
                f"shard={summary['shard_id']:03d} "
                f"samples={summary['samples']:,} "
                f"games={summary['games']:,} "
                f"avg_depth={summary['average_depth']:.2f} "
                f"elapsed={summary['seconds']:.1f}s "
                f"wrote={summary['path']}"
            )
        return

    context = mp.get_context("spawn")
    progress_queue = context.Queue()
    tasks = [
        (
            worker_id,
            first_shard + worker_id,
            sample_target,
            args.seed + iteration * 100_000 + worker_id,
            str(args.data_dir),
            str(model_path),
            args.teacher_states,
            args.root_top_k,
            args.internal_top_k,
            args.search_workers,
            args.extra_random_moves,
            args.best_probability,
            args.top_k_probability,
            args.exploration_top_k,
            args.sample_probability,
            args.max_samples_per_game,
            args.max_game_moves,
            args.progress_every,
        )
        for worker_id, sample_target in enumerate(split_samples(args.samples_per_iteration, workers))
    ]

    print(
        f"Generating {args.samples_per_iteration:,} CNN-guided positions "
        f"with {workers} workers, root_top_k={args.root_top_k}, "
        f"internal_top_k={args.internal_top_k}."
    )
    with context.Pool(workers, initializer=initialize_worker, initargs=(progress_queue,)) as pool:
        results = [pool.apply_async(generate_guided_shard, (task,)) for task in tasks]
        pending = list(results)
        with tqdm(total=args.samples_per_iteration, unit="position", dynamic_ncols=True) as progress:
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


def iteration_from_checkpoint(checkpoint_dir: Path) -> int:
    paths = sorted(checkpoint_dir.glob("iter_*.ts"))
    if not paths:
        return 0
    latest = paths[-1].stem.removeprefix("iter_")
    try:
        return int(latest) + 1
    except ValueError:
        return len(paths)


def main():
    parser = argparse.ArgumentParser(description="Run endless BlitzGo CNN-guided self training.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/move_ranker_7x7"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model/self_train"))
    parser.add_argument("--active-model", type=Path, default=Path("model/move_ranker_self.ts"))
    parser.add_argument("--samples-per-iteration", type=int, default=500_000)
    parser.add_argument("--teacher-states", type=int, default=500_000)
    parser.add_argument("--root-top-k", type=int, default=12)
    parser.add_argument("--internal-top-k", type=int, default=0)
    parser.add_argument("--extra-random-moves", type=int, default=2)
    parser.add_argument("--search-workers", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        help="Self-play processes. Defaults to detected CPUs divided by --search-workers.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=cpu_count())
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--best-probability", type=float, default=0.65)
    parser.add_argument("--top-k-probability", type=float, default=0.25)
    parser.add_argument("--exploration-top-k", type=int, default=5)
    parser.add_argument("--sample-probability", type=float, default=0.50)
    parser.add_argument("--max-samples-per-game", type=int, default=32)
    parser.add_argument("--max-game-moves", type=int, default=BOARD_SIZE * BOARD_SIZE * 4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--start-iteration", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.search_workers = max(1, args.search_workers)
    args.torch_threads = max(1, args.torch_threads)
    if args.workers is not None:
        args.workers = max(1, args.workers)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.active_model.parent.mkdir(parents=True, exist_ok=True)
    iteration = (
        args.start_iteration
        if args.start_iteration is not None
        else iteration_from_checkpoint(args.checkpoint_dir)
    )

    while True:
        model_path = args.checkpoint_dir / f"iter_{iteration:04d}.ts"
        checkpoint_path = args.checkpoint_dir / f"iter_{iteration:04d}.json"
        existing = shard_count(args.data_dir)
        print(
            f"\n=== iteration {iteration} ===\n"
            f"existing_shards={existing} existing_samples={sample_count(args.data_dir):,}"
        )

        if existing:
            run_training(args, iteration, model_path, checkpoint_path)
        else:
            print("No existing data found; saving random CNN bootstrap model.")
            save_random_model(
                model_path,
                args.checkpoint_dir / f"iter_{iteration:04d}_random.pt",
                args.channels,
                args.blocks,
            )
            checkpoint_path.write_text(
                json.dumps(
                    {"iteration": iteration, "random_initialization": True},
                    indent=2,
                )
                + "\n"
            )

        shutil.copyfile(model_path, args.active_model)
        print(f"Active model: {args.active_model}")
        generate_guided_data(args, iteration, model_path)

        iteration += 1
        if args.once:
            break


if __name__ == "__main__":
    main()
