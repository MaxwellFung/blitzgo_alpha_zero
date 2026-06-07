#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm


def digest_board(board: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(board)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.digest()


def shard_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {directory}")
    return paths


def shard_number(path: Path) -> int | None:
    suffix = path.stem.removeprefix("shard_")
    try:
        return int(suffix)
    except ValueError:
        return None


def load_current_boards(paths: list[Path]) -> set[bytes]:
    hashes = set()
    for path in tqdm(paths, desc="Indexing current boards", unit="shard"):
        with np.load(path) as shard:
            for board in shard["boards"]:
                hashes.add(digest_board(board))
    return hashes


def filter_new_rows(path: Path, seen_boards: set[bytes]) -> tuple[dict[str, np.ndarray], int, int]:
    with np.load(path) as shard:
        boards = shard["boards"]
        keep = []
        skipped = 0
        for index, board in enumerate(boards):
            board_hash = digest_board(board)
            if board_hash in seen_boards:
                skipped += 1
                continue
            seen_boards.add(board_hash)
            keep.append(index)

        if not keep:
            return {}, len(boards), skipped

        keep = np.asarray(keep, dtype=np.int64)
        arrays = {
            "boards": boards[keep],
            "legal_masks": shard["legal_masks"][keep],
            "move_scores": shard["move_scores"][keep],
            "completed_depths": shard["completed_depths"][keep],
        }
        return arrays, len(boards), skipped


def confirm() -> bool:
    answer = input("Proceed with writing merged CNN shards? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge old data/cnn_games shards into data/move_ranker_7x7 using "
            "shard_cnnXXX.npz names, skipping board duplicates."
        )
    )
    parser.add_argument("--old-dir", type=Path, default=Path("data/cnn_games"))
    parser.add_argument("--current-dir", type=Path, default=Path("data/move_ranker_7x7"))
    parser.add_argument(
        "--start",
        type=int,
        default=100,
        help="First old numeric shard id to merge. Defaults to 100, skipping duplicate shard_099.",
    )
    parser.add_argument("--end", type=int, default=None, help="Optional inclusive old shard id to stop at.")
    parser.add_argument("--prefix", default="shard_cnn", help="Output file prefix.")
    parser.add_argument("--yes", action="store_true", help="Write without asking for confirmation.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard_cnn*.npz outputs if they already exist.",
    )
    args = parser.parse_args()

    current_paths = shard_paths(args.current_dir)
    old_paths = []
    for path in shard_paths(args.old_dir):
        number = shard_number(path)
        if number is None or number < args.start:
            continue
        if args.end is not None and number > args.end:
            continue
        old_paths.append(path)

    if not old_paths:
        raise FileNotFoundError(
            f"No old numeric shards found in {args.old_dir} for range {args.start}..{args.end or 'end'}"
        )

    outputs = [args.current_dir / f"{args.prefix}{shard_number(path):03d}.npz" for path in old_paths]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing merged shards. "
            f"First existing file: {existing[0]}. Pass --overwrite if intentional."
        )

    print(f"Current folder: {args.current_dir} ({len(current_paths):,} shards)")
    print(f"Old CNN folder: {args.old_dir} ({len(old_paths):,} shards selected)")
    print(f"Output naming: {args.current_dir}/{args.prefix}XXX.npz")
    print(f"Selected old range: shard_{shard_number(old_paths[0]):03d}.npz .. shard_{shard_number(old_paths[-1]):03d}.npz")
    if not args.yes and not confirm():
        print("Cancelled; no shards were merged.")
        return

    seen_boards = load_current_boards(current_paths)
    total_input = 0
    total_written = 0
    total_skipped = 0
    written_files = 0

    for old_path, output_path in tqdm(list(zip(old_paths, outputs)), desc="Merging old shards", unit="shard"):
        arrays, input_count, skipped = filter_new_rows(old_path, seen_boards)
        total_input += input_count
        total_skipped += skipped
        if not arrays:
            tqdm.write(f"{old_path.name}: skipped all {input_count:,} rows")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **arrays)
        written = len(arrays["boards"])
        total_written += written
        written_files += 1
        tqdm.write(
            f"{old_path.name} -> {output_path.name}: "
            f"wrote={written:,} skipped_duplicates={skipped:,}"
        )

    print("\nMerge summary")
    print(f"  Old rows scanned: {total_input:,}")
    print(f"  Rows written: {total_written:,}")
    print(f"  Duplicate board rows skipped: {total_skipped:,}")
    print(f"  Output shard files written: {written_files:,}")


if __name__ == "__main__":
    main()
