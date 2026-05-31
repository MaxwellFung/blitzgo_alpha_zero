#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class MoveRanker(nn.Module):
    def __init__(self, channels: int = 96, blocks: int = 6):
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

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        return self.policy(self.body(self.stem(board.float())))


def encode_game(game) -> np.ndarray:
    size = game.size()
    if size != BOARD_SIZE:
        raise ValueError(f"Move ranker expects {BOARD_SIZE}x{BOARD_SIZE}, got {size}x{size}")

    stones = np.asarray(game.stones(), dtype=np.uint8).reshape(size, size)
    territories = np.asarray(game.territories(), dtype=np.uint8).reshape(size, size)
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


def ranked_moves(model, game) -> list[int]:
    board = torch.from_numpy(encode_game(game)).unsqueeze(0)
    with torch.no_grad():
        logits = model(board).squeeze(0).cpu().numpy()
    return sorted(game.legal_moves_all(), key=lambda move: (-float(logits[move]), move))


def load_scripted_model(path: str | Path):
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return model


def load_checkpoint_model(path: str | Path):
    checkpoint = torch.load(str(path), map_location="cpu")
    model = MoveRanker(
        channels=int(checkpoint.get("channels", 96)),
        blocks=int(checkpoint.get("blocks", 6)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
