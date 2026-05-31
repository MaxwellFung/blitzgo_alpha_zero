#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from move_ranker import ACTION_SIZE, BOARD_SIZE, CHANNELS, MoveRanker


def load_shards(data_dir: str):
    paths = sorted(Path(data_dir).glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {data_dir}")

    shards = [np.load(path) for path in paths]
    boards = np.concatenate([shard["boards"] for shard in shards])
    legal_masks = np.concatenate([shard["legal_masks"] for shard in shards])
    move_scores = np.concatenate([shard["move_scores"] for shard in shards])
    return boards, legal_masks, move_scores


def ranking_loss(logits, legal_masks, move_scores, temperature: float):
    masked_logits = logits.masked_fill(~legal_masks, -1e9)
    masked_scores = move_scores.float().masked_fill(~legal_masks, -1e9)
    targets = F.softmax(masked_scores / temperature, dim=1)
    return -(targets * F.log_softmax(masked_logits, dim=1)).sum(dim=1).mean()


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


def main():
    parser = argparse.ArgumentParser(description="Train the BlitzGo CNN move ranker.")
    parser.add_argument("--data-dir", default="data/move_ranker_7x7")
    parser.add_argument("--output", default="model/move_ranker_7x7.ts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
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
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    model = MoveRanker(channels=args.channels, blocks=args.blocks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val_loss = float("inf")
    best_state = None

    print(f"Training on {train_size:,} positions; validating on {val_size:,}.")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for board, mask, scores in train_loader:
            if not args.no_augment:
                board, mask, scores = augment_batch(board, mask, scores)
            optimizer.zero_grad()
            loss = ranking_loss(model(board), mask, scores, args.temperature)
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * len(board)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_top1 = 0.0
        val_top3 = 0.0
        val_top5 = 0.0
        with torch.no_grad():
            for board, mask, scores in val_loader:
                logits = model(board)
                val_loss += float(ranking_loss(logits, mask, scores, args.temperature)) * len(board)
                val_top1 += float(topk_accuracy(logits, mask, scores, 1)) * len(board)
                val_top3 += float(topk_accuracy(logits, mask, scores, 3)) * len(board)
                val_top5 += float(topk_accuracy(logits, mask, scores, 5)) * len(board)

        epoch_val_loss = val_loss / val_size
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss / train_size:.4f} "
            f"val_loss={epoch_val_loss:.4f} "
            f"val_top1={val_top1 / val_size:.3f} "
            f"val_top3={val_top3 / val_size:.3f} "
            f"val_top5={val_top5 / val_size:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.trace(model.eval(), torch.zeros(1, CHANNELS, BOARD_SIZE, BOARD_SIZE))
    scripted.save(str(output))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
