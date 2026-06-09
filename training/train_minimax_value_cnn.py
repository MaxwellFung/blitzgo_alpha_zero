#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


BOARD_SIZE = 7
CHANNELS = 5
VALUE_SCALE = 500.0


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class MinimaxValueCNN(nn.Module):
    def __init__(self, channels: int = 96, blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.value = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = self.stem(board.float())
        x = self.body(x)
        return self.value(x).squeeze(1)


def load_data(path: Path):
    with np.load(path) as data:
        if "boards" not in data or "targets" not in data:
            raise ValueError(f"{path} must contain 'boards' and 'targets'.")
        return data["boards"], data["targets"].astype(np.float32)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available in this PyTorch build.")
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    total = 0
    loss_sum = 0.0
    abs_sum = 0.0
    sign_correct = 0
    with torch.no_grad():
        for board, target in loader:
            board = board.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model(board)
            loss_sum += float(F.mse_loss(prediction, target, reduction="sum"))
            abs_sum += float((prediction - target).abs().sum())
            sign_correct += int((torch.sign(prediction) == torch.sign(target)).sum())
            total += len(board)
    return loss_sum / total, abs_sum / total, sign_correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Train a larger CNN to predict minimax value from board state."
    )
    parser.add_argument("--data", type=Path, default=Path("data/compiled/value_7x7.npz"))
    parser.add_argument("--checkpoint", type=Path, default=Path("model/minimax_value_cnn.pt"))
    parser.add_argument("--torchscript-output", type=Path, default=Path("model/minimax_value_cnn.ts"))
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=max(1, torch.get_num_threads()))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    device = pick_device(args.device)

    boards, targets = load_data(args.data)
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

    model = MinimaxValueCNN(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_mse = float("inf")
    best_state = None
    print(
        f"Loaded {len(dataset):,} positions from {args.data}. "
        f"Training {train_size:,}, validating {val_size:,}. Device={device}."
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

        val_mse, val_mae, sign_acc = evaluate(model, val_loader, device)
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} train_mse={train_loss / train_size:.6f} "
            f"val_mse={val_mse:.6f} val_mae={val_mae:.4f} "
            f"score_mae={val_mae * VALUE_SCALE:.1f} sign_acc={sign_acc:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu().eval()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "channels": args.channels,
            "blocks": args.blocks,
            "value_scale": VALUE_SCALE,
        },
        args.checkpoint,
    )
    args.torchscript_output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.trace(model, torch.zeros(1, CHANNELS, BOARD_SIZE, BOARD_SIZE)).save(
        str(args.torchscript_output)
    )
    print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Saved TorchScript CNN: {args.torchscript_output}")
    print(f"Best val_mse: {best_mse:.6f}")


if __name__ == "__main__":
    main()
