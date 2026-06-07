#!/usr/bin/env python3
import argparse
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from value_ranker import VALUE_SCALE


ILLEGAL_SCORE = np.iinfo(np.int16).min


def format_bytes(count: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(count)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def array_summary(name: str, array: np.ndarray) -> str:
    return (
        f"  {name}: shape={array.shape}, dtype={array.dtype}, "
        f"size={format_bytes(array.nbytes)}"
    )


def confirm_proceed() -> bool:
    answer = input("Proceed with writing compiled datasets? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def is_cnn_shard(path: Path) -> bool:
    return path.stem.startswith("shard_cnn")


def load_move_shards(data_dir: Path):
    paths = sorted(data_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {data_dir}")

    boards = []
    legal_masks = []
    move_scores = []
    completed_depths = []
    for path in tqdm(paths, desc="Loading move shards", unit="shard"):
        with np.load(path) as shard:
            boards.append(shard["boards"])
            legal_masks.append(shard["legal_masks"])
            move_scores.append(shard["move_scores"])
            if "completed_depths" in shard:
                completed_depths.append(shard["completed_depths"])

    boards = np.concatenate(boards)
    legal_masks = np.concatenate(legal_masks)
    move_scores = np.concatenate(move_scores)
    if completed_depths:
        completed_depths = np.concatenate(completed_depths)
    else:
        completed_depths = np.zeros(len(boards), dtype=np.int16)
    return paths, boards, legal_masks, move_scores, completed_depths


def source_position_counts(paths: list[Path]) -> tuple[int, int, int, int]:
    regular_shards = 0
    regular_positions = 0
    cnn_shards = 0
    cnn_positions = 0
    for path in paths:
        with np.load(path) as shard:
            positions = len(shard["boards"])
        if is_cnn_shard(path):
            cnn_shards += 1
            cnn_positions += positions
        else:
            regular_shards += 1
            regular_positions += positions
    return regular_shards, regular_positions, cnn_shards, cnn_positions


def load_completed_game_values(game_data_dir: Path):
    if not game_data_dir.exists():
        return None, None, 0

    boards = []
    targets = []
    skipped = 0
    for path in tqdm(sorted(game_data_dir.glob("game_*.npz")), desc="Loading UI games", unit="game"):
        try:
            with np.load(path) as game:
                if not bool(game["completed"][0]):
                    continue
                outcome_targets = game["outcome_targets"].astype(np.float32)
                usable = np.isfinite(outcome_targets)
                if usable.any():
                    boards.append(game["boards"][usable])
                    targets.append(outcome_targets[usable])
        except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
            skipped += 1
            tqdm.write(f"Skipping unreadable UI game {path}: {error}")

    if not boards:
        return None, None, skipped
    return np.concatenate(boards), np.concatenate(targets), skipped


def write_npz(path: Path, compress: bool, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if compress else np.savez
    saver(path, **arrays)


def main():
    parser = argparse.ArgumentParser(
        description="Compile BlitzGo training shards into combined policy and value datasets."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/move_ranker_7x7"))
    parser.add_argument("--game-data-dir", type=Path, default=Path("data/ui_games"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/compiled"))
    parser.add_argument("--policy-output", type=Path, default=None)
    parser.add_argument("--value-output", type=Path, default=None)
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Write compiled datasets without asking for confirmation.",
    )
    parser.add_argument(
        "--skip-ui-games",
        action="store_true",
        help="Do not add completed UI-game outcome targets to the value dataset.",
    )
    args = parser.parse_args()

    policy_output = args.policy_output or args.output_dir / "policy_topk_7x7.npz"
    value_output = args.value_output or args.output_dir / "value_7x7.npz"
    compress = not args.no_compress

    paths, boards, legal_masks, move_scores, completed_depths = load_move_shards(args.data_dir)
    regular_shards, regular_positions, cnn_shards, cnn_positions = source_position_counts(paths)
    masked_scores = np.where(legal_masks, move_scores, ILLEGAL_SCORE)
    best_scores = masked_scores.max(axis=1).astype(np.float32)
    best_moves = masked_scores.argmax(axis=1).astype(np.int16)
    value_targets = np.tanh(best_scores / VALUE_SCALE).astype(np.float32)
    self_play_value_positions = len(boards)
    ui_game_value_positions = 0

    source_shards = np.asarray([str(path) for path in paths])
    print(f"Loaded {len(boards):,} move positions from {len(paths):,} shards.")

    value_boards = boards
    if not args.skip_ui_games:
        game_boards, game_targets, skipped_games = load_completed_game_values(args.game_data_dir)
        if skipped_games:
            print(f"Skipped {skipped_games:,} unreadable UI-game file(s).")
        if game_boards is not None:
            ui_game_value_positions = len(game_boards)
            value_boards = np.concatenate((value_boards, game_boards))
            value_targets = np.concatenate((value_targets, game_targets.astype(np.float32)))
            print(f"Added {ui_game_value_positions:,} completed UI-game value positions.")

    policy_bytes = (
        boards.nbytes
        + legal_masks.nbytes
        + move_scores.nbytes
        + completed_depths.nbytes
        + best_moves.nbytes
        + best_scores.nbytes
    )
    value_bytes = value_boards.nbytes + value_targets.nbytes
    print("\nDataset summary")
    print(f"  Source move shards: {len(paths):,}")
    print(f"    Regular shard files: {regular_shards:,}")
    print(f"    Merged CNN shard files: {cnn_shards:,}")
    print(f"  Policy positions: {len(boards):,}")
    print(f"    Regular shard policy positions: {regular_positions:,}")
    print(f"    Merged CNN policy positions: {cnn_positions:,}")
    print(f"  Value positions: {len(value_boards):,}")
    print(f"    Self-play minimax value positions: {self_play_value_positions:,}")
    print(f"      Regular shard value positions: {regular_positions:,}")
    print(f"      Merged CNN shard value positions: {cnn_positions:,}")
    print(f"    Completed UI-game value positions: {ui_game_value_positions:,}")
    print(f"  Policy output: {policy_output}")
    print(f"  Value output: {value_output}")
    print(f"  Compression: {'on' if compress else 'off'}")
    print(f"  Approx policy array bytes before compression: {format_bytes(policy_bytes)}")
    print(f"  Approx value array bytes before compression: {format_bytes(value_bytes)}")
    print("  Policy arrays:")
    print(array_summary("boards", boards))
    print(array_summary("legal_masks", legal_masks))
    print(array_summary("move_scores", move_scores))
    print(array_summary("completed_depths", completed_depths))
    print(array_summary("best_moves", best_moves))
    print(array_summary("best_scores", best_scores))
    print("  Value arrays:")
    print(array_summary("boards", value_boards))
    print(array_summary("targets", value_targets))

    if not args.yes and not confirm_proceed():
        print("Cancelled; no compiled datasets were written.")
        return

    print(f"Writing policy dataset: {policy_output}")
    write_npz(
        policy_output,
        compress,
        boards=boards,
        legal_masks=legal_masks,
        move_scores=move_scores,
        completed_depths=completed_depths,
        best_moves=best_moves,
        best_scores=best_scores,
        regular_shard_positions=np.asarray([regular_positions], dtype=np.int64),
        cnn_shard_positions=np.asarray([cnn_positions], dtype=np.int64),
        source_shards=source_shards,
    )

    print(f"Writing value dataset: {value_output}")
    write_npz(
        value_output,
        compress,
        boards=value_boards,
        targets=value_targets,
        self_play_positions=np.asarray([self_play_value_positions], dtype=np.int64),
        ui_game_positions=np.asarray([ui_game_value_positions], dtype=np.int64),
        regular_shard_positions=np.asarray([regular_positions], dtype=np.int64),
        cnn_shard_positions=np.asarray([cnn_positions], dtype=np.int64),
        source_shards=source_shards,
    )
    print(
        f"Done. Policy positions={len(boards):,}; "
        f"value positions={len(value_boards):,} "
        f"({self_play_value_positions:,} self-play, {ui_game_value_positions:,} UI-game)."
    )


if __name__ == "__main__":
    main()
