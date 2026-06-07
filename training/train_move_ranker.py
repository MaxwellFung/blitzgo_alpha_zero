# policy cnn

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


BOARD_SIZE = 7
CHANNELS = 5
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class MoveRanker(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, ACTION_SIZE),
        )

    def forward(self, board):
        return self.policy(self.body(self.stem(board.float())))


def load_shards(data_dir: str):
    data_path = Path(data_dir)
    if data_path.is_file():
        with np.load(data_path) as data:
            return data["boards"], data["legal_masks"], data["move_scores"]

    paths = sorted(data_path.glob("shard_*.npz"))
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
    return boards, legal_masks, move_scores


def ranking_loss(logits, legal_masks, move_scores, temperature: float):
    masked_logits = logits.masked_fill(~legal_masks, torch.finfo(logits.dtype).min)
    masked_scores = move_scores.float().masked_fill(~legal_masks, torch.finfo(logits.dtype).min)

    targets = F.softmax(masked_scores / temperature, dim=1)
    log_probs = F.log_softmax(masked_logits, dim=1)

    return F.kl_div(log_probs, targets, reduction="batchmean", log_target=False)


def topk_accuracy(logits, legal_masks, move_scores, k: int):
    masked_logits = logits.masked_fill(~legal_masks, torch.finfo(logits.dtype).min)
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
    transform = int(torch.randint(0, 8, device=board.device, size=()).item())
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


def train(
    data_dir="blitzgo_alpha_zero/data/move_ranker",
    output="blitzgo_alpha_zero/models/move_ranker.ts",
    checkpoint="blitzgo_alpha_zero/models/move_ranker.pt",
    epochs=40,
    batch_size=1024,
    channels=96,
    blocks=6,
    learning_rate=3e-4,
    temperature=1.0,
    seed=1234,
    augment=True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    boards, legal_masks, move_scores = load_shards(data_dir)
    print(f"Loaded {len(boards):,} positions from {data_dir}")

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
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = MoveRanker(channels=channels, blocks=blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    print(f"Training on {train_size:,}; validating on {val_size:,}.")
    print(f"Model: channels={channels}, blocks={blocks}, batch_size={batch_size}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for board, mask, scores in train_loader:
            board = board.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)

            if augment:
                board, mask, scores = augment_batch(board, mask, scores)

            optimizer.zero_grad(set_to_none=True)
            loss = ranking_loss(model(board), mask, scores, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                loss = ranking_loss(logits, mask, scores, temperature)

                val_loss += float(loss.detach()) * len(board)
                val_top1 += float(topk_accuracy(logits, mask, scores, 1)) * len(board)
                val_top3 += float(topk_accuracy(logits, mask, scores, 3)) * len(board)
                val_top5 += float(topk_accuracy(logits, mask, scores, 5)) * len(board)
                val_top10 += float(topk_accuracy(logits, mask, scores, 10)) * len(board)

        train_loss /= train_size
        val_loss /= val_size
        val_top1 /= val_size
        val_top3 /= val_size
        val_top5 /= val_size
        val_top10 /= val_size

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_top1={val_top1:.3f} "
            f"val_top3={val_top3:.3f} "
            f"val_top5={val_top5:.3f} "
            f"val_top10={val_top10:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    output = Path(output)
    checkpoint = Path(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.cpu().eval()

    torch.save(
        {
            "model_state_dict": model_cpu.state_dict(),
            "channels": channels,
            "blocks": blocks,
            "board_size": BOARD_SIZE,
            "best_val_loss": best_val_loss,
        },
        checkpoint,
    )

    scripted = torch.jit.trace(
        model_cpu,
        torch.zeros(1, CHANNELS, BOARD_SIZE, BOARD_SIZE),
    )
    scripted.save(str(output))

    print(f"Saved TorchScript model: {output}")
    print(f"Saved checkpoint: {checkpoint}")
    print(f"Best val_loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train the BlitzGo policy move-ranker CNN.")
    parser.add_argument("--data-file", default="data/compiled/policy_topk_7x7.npz")
    parser.add_argument("--output", default="model/move_ranker.ts")
    parser.add_argument("--checkpoint", default="model/move_ranker.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    train(
        data_dir=args.data_file,
        output=args.output,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        channels=args.channels,
        blocks=args.blocks,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        augment=not args.no_augment,
    )


if __name__ == "__main__":
    main()
