#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py"))

from value_ranker import TinyValueRanker, VALUE_SCALE, export_native_model


ILLEGAL_SCORE = np.iinfo(np.int16).min


def load_shards(data_dir: Path, game_data_dir: Path | None = None):
    if data_dir.is_file():
        with np.load(data_dir) as data:
            if "targets" not in data:
                raise ValueError(
                    f"{data_dir} is not a compiled value dataset; missing 'targets'."
                )
            return data["boards"], data["targets"], 0

    paths = sorted(data_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {data_dir}")
    boards = []
    legal_masks = []
    move_scores = []
    for path in paths:
        with np.load(path) as shard:
            boards.append(shard["boards"])
            legal_masks.append(shard["legal_masks"])
            move_scores.append(shard["move_scores"])
    boards = np.concatenate(boards)
    legal_masks = np.concatenate(legal_masks)
    move_scores = np.concatenate(move_scores)
    masked_scores = np.where(legal_masks, move_scores, ILLEGAL_SCORE)
    best_scores = masked_scores.max(axis=1).astype(np.float32)
    targets = np.tanh(best_scores / VALUE_SCALE).astype(np.float32)
    game_positions = 0
    if game_data_dir is not None and game_data_dir.exists():
        game_boards = []
        game_targets = []
        for path in sorted(game_data_dir.glob("game_*.npz")):
            with np.load(path) as game:
                if not bool(game["completed"][0]):
                    continue
                outcome_targets = game["outcome_targets"].astype(np.float32)
                usable = np.isfinite(outcome_targets)
                if usable.any():
                    game_boards.append(game["boards"][usable])
                    game_targets.append(outcome_targets[usable])
        if game_boards:
            added_boards = np.concatenate(game_boards)
            added_targets = np.concatenate(game_targets)
            game_positions = len(added_boards)
            boards = np.concatenate((boards, added_boards))
            targets = np.concatenate((targets, added_targets))
    return boards, targets, game_positions


def main():
    parser = argparse.ArgumentParser(description="Train the tiny native BlitzGo value CNN.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/compiled/value_7x7.npz"))
    parser.add_argument("--game-data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("model/value_ranker.bin"))
    parser.add_argument("--torchscript-output", type=Path, default=Path("model/value_ranker.ts"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=max(1, torch.get_num_threads()))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boards, targets, game_positions = load_shards(args.data_dir, args.game_data_dir)
    dataset = TensorDataset(torch.from_numpy(boards), torch.from_numpy(targets))
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
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
    model = TinyValueRanker().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_loss = float("inf")
    best_state = None
    print(
        f"Loaded {len(dataset):,} positions. Training on {train_size:,}; "
        f"validating on {val_size:,}. Included {game_positions:,} completed UI-game "
        f"positions. Device={device}; torch_threads={args.torch_threads}."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for board, target in train_loader:
            board = board.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(board), target)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(board)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for board, target in val_loader:
                board = board.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                val_loss += float(F.mse_loss(model(board), target)) * len(board)
        val_loss /= val_size
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} train_mse={train_loss / train_size:.6f} "
            f"val_mse={val_loss:.6f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()
    export_native_model(model, args.output)
    args.torchscript_output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.trace(model, torch.zeros(1, 5, 7, 7))
    scripted.save(str(args.torchscript_output))
    print(f"Saved native value model: {args.output}")
    print(f"Saved TorchScript reference model: {args.torchscript_output}")
    print(f"Best val_mse: {best_loss:.6f}")


if __name__ == "__main__":
    main()
