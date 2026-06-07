#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm


def shard_id(path: Path) -> int:
    suffix = path.stem.removeprefix("shard_")
    if suffix.startswith("cnn"):
        suffix = suffix.removeprefix("cnn")
    return int(suffix)


def shard_paths(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("shard_*.npz"), key=shard_id)
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {directory}")
    return paths


def digest_parts(*arrays) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.digest()


def shard_hashes(path: Path) -> tuple[set[bytes], set[bytes], int]:
    with np.load(path) as shard:
        boards = shard["boards"]
        legal_masks = shard["legal_masks"]
        move_scores = shard["move_scores"]
        completed_depths = shard["completed_depths"]

        board_hashes = set()
        exact_hashes = set()
        for index in range(len(boards)):
            board_hashes.add(digest_parts(boards[index]))
            exact_hashes.add(
                digest_parts(
                    boards[index],
                    legal_masks[index],
                    move_scores[index],
                    completed_depths[index],
                )
            )
        return board_hashes, exact_hashes, len(boards)


def build_current_index(paths: list[Path]) -> tuple[set[bytes], set[bytes], int]:
    board_hashes = set()
    exact_hashes = set()
    positions = 0
    for path in tqdm(paths, desc="Indexing current shards", unit="shard"):
        shard_boards, shard_exact, shard_positions = shard_hashes(path)
        board_hashes.update(shard_boards)
        exact_hashes.update(shard_exact)
        positions += shard_positions
    return board_hashes, exact_hashes, positions


def overlap_report(path: Path, current_boards: set[bytes], current_exact: set[bytes]) -> dict:
    board_hashes, exact_hashes, positions = shard_hashes(path)
    board_overlap = len(board_hashes & current_boards)
    exact_overlap = len(exact_hashes & current_exact)
    unique_boards = len(board_hashes)
    unique_exact = len(exact_hashes)
    return {
        "path": path,
        "positions": positions,
        "unique_boards": unique_boards,
        "unique_exact": unique_exact,
        "board_overlap": board_overlap,
        "exact_overlap": exact_overlap,
        "board_overlap_percent": 100.0 * board_overlap / max(1, unique_boards),
        "exact_overlap_percent": 100.0 * exact_overlap / max(1, unique_exact),
    }


def print_report(report: dict):
    print(
        f"{report['path']}: positions={report['positions']:,} "
        f"unique_boards={report['unique_boards']:,} "
        f"board_overlap={report['board_overlap']:,} "
        f"({report['board_overlap_percent']:.2f}%) "
        f"exact_overlap={report['exact_overlap']:,} "
        f"({report['exact_overlap_percent']:.2f}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether old cnn_games shards overlap current move_ranker_7x7 "
            "data before merging."
        )
    )
    parser.add_argument("--old-dir", type=Path, default=Path("data/cnn_games"))
    parser.add_argument("--current-dir", type=Path, default=Path("data/move_ranker_7x7"))
    parser.add_argument("--first-shard", default="shard_099.npz")
    parser.add_argument(
        "--stop-on-overlap",
        action="store_true",
        help="Exit with status 2 if any old shard overlaps current data by board.",
    )
    args = parser.parse_args()

    old_paths = shard_paths(args.old_dir)
    current_paths = shard_paths(args.current_dir)
    first_path = args.old_dir / args.first_shard
    if not first_path.exists():
        raise FileNotFoundError(f"Missing first overlap check shard: {first_path}")

    print(
        f"Current data: {args.current_dir} ({len(current_paths):,} shard files)\n"
        f"Old data: {args.old_dir} ({len(old_paths):,} shard files)"
    )
    current_boards, current_exact, current_positions = build_current_index(current_paths)
    print(
        f"Indexed current data: positions={current_positions:,} "
        f"unique_boards={len(current_boards):,} unique_exact_samples={len(current_exact):,}"
    )

    print(f"\nFirst required check: {first_path}")
    first_report = overlap_report(first_path, current_boards, current_exact)
    print_report(first_report)

    print("\nChecking every old shard against current data")
    reports = []
    for path in tqdm(old_paths, desc="Checking old shards", unit="shard"):
        report = overlap_report(path, current_boards, current_exact)
        reports.append(report)
        print_report(report)

    overlapping = [report for report in reports if report["board_overlap"] > 0]
    exact_overlapping = [report for report in reports if report["exact_overlap"] > 0]
    total_positions = sum(report["positions"] for report in reports)
    total_board_overlap = sum(report["board_overlap"] for report in reports)
    total_exact_overlap = sum(report["exact_overlap"] for report in reports)

    print("\nSummary")
    print(f"  Old shards checked: {len(reports):,}")
    print(f"  Old positions checked: {total_positions:,}")
    print(f"  Shards with board overlap: {len(overlapping):,}")
    print(f"  Shards with exact-sample overlap: {len(exact_overlapping):,}")
    print(f"  Total unique-board overlaps counted per shard: {total_board_overlap:,}")
    print(f"  Total exact-sample overlaps counted per shard: {total_exact_overlap:,}")
    if overlapping:
        print("  Overlapping shards:")
        for report in overlapping:
            print(
                f"    {report['path'].name}: board_overlap={report['board_overlap']:,} "
                f"exact_overlap={report['exact_overlap']:,}"
            )
    else:
        print("  No board overlap found. Old shards look safe to append after current data.")

    if args.stop_on_overlap and overlapping:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
