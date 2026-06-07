#!/usr/bin/env python3
import struct
from pathlib import Path

import torch
import torch.nn as nn


BOARD_SIZE = 7
CHANNELS = 5
VALUE_CHANNELS = 8
VALUE_SCALE = 500.0


class TinyValueRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, VALUE_CHANNELS, 3, padding=1)
        self.conv2 = nn.Conv2d(VALUE_CHANNELS, VALUE_CHANNELS, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Linear(VALUE_CHANNELS * BOARD_SIZE * BOARD_SIZE, 1)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = board.float()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        return torch.tanh(self.output(x)).squeeze(1)


def export_native_model(model: TinyValueRanker, path: str | Path):
    model = model.cpu().eval()
    state = model.state_dict()
    tensors = (
        state["conv1.weight"],
        state["conv1.bias"],
        state["conv2.weight"],
        state["conv2.bias"],
        state["output.weight"],
        state["output.bias"],
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(b"BLITZVAL1")
        output.write(struct.pack("<f", VALUE_SCALE))
        for tensor in tensors:
            values = tensor.detach().contiguous().view(-1).tolist()
            output.write(struct.pack(f"<{len(values)}f", *values))
