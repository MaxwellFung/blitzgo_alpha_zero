#!/usr/bin/env python3
# ============================================================
# Play against trained AlphaZero model using:
#   - C++ Game from az_engine
#   - C++ MCTS with NN eval callback
#
# AUTO-CONFIG:
#   - channels, mcts_sims, c_puct loaded from checkpoint
#   - backward compatible with old checkpoints
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import az_engine

# ================== CONFIG ==================

BOARD_SIZE = 3
DEVICE = "cpu"   # "cuda" / "mps"

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = f"az_territory_size{BOARD_SIZE}.pt"

# ============================================


def resolve_device(dev: str) -> str:
    if dev == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return dev


DEVICE = resolve_device(DEVICE)


# ============================ NEURAL NET ============================

class AZNet(nn.Module):
    def __init__(self, board_size: int, channels: int):
        super().__init__()
        self.board_size = board_size
        self.action_dim = board_size * board_size

        self.conv1 = nn.Conv2d(5, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)

        self.p_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.ReLU(),
        )
        self.p_fc = nn.Linear(2 * board_size * board_size, self.action_dim)

        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.ReLU(),
        )
        self.v_fc1 = nn.Linear(board_size * board_size, channels)
        self.v_fc2 = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        p = self.p_head(x).flatten(1)
        p = self.p_fc(p)

        v = self.v_head(x).flatten(1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))

        return p, v.squeeze(-1)


def make_eval_fn(model: AZNet):
    model.eval()

    @torch.no_grad()
    def eval_fn(st):
        x = torch.from_numpy(np.array(st, copy=False)).to(DEVICE)
        if x.ndim == 3:
            x = x.unsqueeze(0)

        logits, v = model(x)
        priors = (
            torch.softmax(logits, dim=-1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        value = float(v.squeeze(0).cpu().item())
        return priors, value

    return eval_fn


# ============================ CHECKPOINT CONFIG ============================

def load_config_from_ckpt(ckpt: dict):
    sd = ckpt["state_dict"]

    # Infer channels directly from conv1 weight
    channels = sd["conv1.weight"].shape[0]

    return {
        "channels": channels,
        "mcts_sims": ckpt.get("mcts_sims", 100),
        "c_puct": ckpt.get("c_puct", 1.5),
    }


# ============================ UTILS ============================

def print_board(game: az_engine.Game, size: int):
    st = np.array(game.state_tensor(), copy=False)  # (5, N, N)

    stones_p1 = st[0]
    stones_p2 = st[1]
    terr_p1 = st[2]
    terr_p2 = st[3]

    print("\nBoard:")
    for y in range(size):
        row = []
        for x in range(size):
            if stones_p1[y, x]:
                row.append("X")
            elif stones_p2[y, x]:
                row.append("O")
            elif terr_p1[y, x]:
                row.append("x")
            elif terr_p2[y, x]:
                row.append("o")
            else:
                row.append(".")
        print(" ".join(row))
    print()


def ask_human_move(game: az_engine.Game, size: int) -> int:
    legal = set(game.legal_moves_all())
    while True:
        s = input("Your move (x y) [1-based]: ").strip()
        try:
            x1, y1 = map(int, s.split())
            x = x1 - 1
            y = y1 - 1
            if not (0 <= x < size and 0 <= y < size):
                print("Out of bounds.")
                continue
            a = y * size + x
            if a in legal:
                return a
            else:
                print("Illegal move.")
        except Exception:
            print("Enter: x y   (both from 1 to N)")


# ============================ MAIN ============================

def main():
    size = BOARD_SIZE
    ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = load_config_from_ckpt(ckpt)

    channels = cfg["channels"]
    mcts_sims = cfg["mcts_sims"]
    c_puct = cfg["c_puct"]

    print(
        f"Loaded checkpoint config | "
        f"channels={channels}, "
        f"mcts_sims={mcts_sims}, "
        f"c_puct={c_puct}"
    )

    model = AZNet(
        board_size=ckpt["board_size"],
        channels=channels,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)

    eval_fn = make_eval_fn(model)

    mcts = az_engine.MCTS(
        cpuct=c_puct,
        n_sims=mcts_sims,
        dir_alpha=0.0,
        dir_eps=0.0,
        seed=1234,
    )

    game = az_engine.Game(size)

    print("Play against model.")
    side = input("Play as player 1 (X) or player 2 (O)? [1/2]: ").strip()
    human_side = 1 if side != "2" else 2

    while not game.is_over():
        print_board(game, size)
        stm = game.current_player()

        if stm == human_side:
            a = ask_human_move(game, size)
            game.apply(a)
        else:
            policy, _ = mcts.run(game, eval_fn)
            policy = np.asarray(policy).reshape(-1)

            if policy.sum() <= 0:
                legal = game.legal_moves_all()
                a = int(np.random.choice(legal))
            else:
                a = int(np.argmax(policy))

            game.apply(a)
            print(f"Model plays: ({a % size + 1}, {a // size + 1})")

    print_board(game, size)
    w = game.winner()
    if w == 0:
        print("Draw.")
    elif w == human_side:
        print("You win! 🎉")
    else:
        print("Model wins.")


if __name__ == "__main__":
    main()