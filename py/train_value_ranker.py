#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from value_ranker import TinyValueRanker, VALUE_SCALE, export_native_model


ILLEGAL_SCORE = np.iinfo(np.int16).min


def load_shards(data_dir: Path):
    paths = sorted(data_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {data_dir}")
    shards = [np.load(path) for path in paths]
    boards = np.concatenate([shard["boards"] for shard in shards])
    legal_masks = np.concatenate([shard["legal_masks"] for shard in shards])
    move_scores = np.concatenate([shard["move_scores"] for shard in shards])
    masked_scores = np.where(legal_masks, move_scores, ILLEGAL_SCORE)
    best_scores = masked_scores.max(axis=1).astype(np.float32)
    targets = np.tanh(best_scores / VALUE_SCALE).astype(np.float32)
    return boards, targets


def main():
    parser = argparse.ArgumentParser(description="Train the tiny native BlitzGo value CNN.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/move_ranker_7x7"))
    parser.add_argument("--output", type=Path, default=Path("model/value_ranker.bin"))
    parser.add_argument("--torchscript-output", type=Path, default=Path("model/value_ranker.ts"))
    parser.add_argument("--epochs", type=int, default=20)
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
    boards, targets = load_shards(args.data_dir)
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
        f"validating on {val_size:,}. Device={device}; torch_threads={args.torch_threads}."
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
