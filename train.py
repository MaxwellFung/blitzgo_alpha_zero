#!/usr/bin/env python3
# ============================================================
# AlphaZero-like training loop using:
#   - C++ Game (PyGame) from az_engine
#   - C++ MCTS from az_engine (Python NN eval callback)
# Notes:
#   - The C++ MCTS.run(game, eval_fn) returns (policy, best_move)
#   - policy is a length n*n float vector already normalized
#   - We do NOT clone game; we mutate/undo in-place
#
# MULTIPROCESSING:
#   - Parallel self-play + evaluation on CPU
#   - Each worker holds its own model copies (safe w/ spawn)
#   - If DEVICE != "cpu", workers are forced to 1 (no GPU sharing)
# ============================================================

import os, random
import multiprocessing as mp
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import az_engine  # compiled pybind module

# ================== HYPERPARAMETERS ==================

BOARD_SIZE = 3

N_ITERS = 50
N_SELFPLAY_GAMES = 50
MCTS_SIMS = 100

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.5
DIRICHLET_EPS = 0.25

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS = 5
REPLAY_BUFFER_SIZE = 5000

TEMPERATURE_MOVES = 5
TEMPERATURE = 1.0

EVAL_GAMES = 20
ACCEPT_WINRATE = 0.55
CHANNELS = 196

SEED = 42
DEVICE = "cpu"  # "cuda" / "mps"

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = f"az_territory_size{BOARD_SIZE}.pt"

# =====================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(dev: str) -> str:
    if dev == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return dev


set_seed(SEED)
DEVICE = resolve_device(DEVICE)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================ NEURAL NET ============================


class AZNet(nn.Module):
    def __init__(self, board_size: int, channels: int = CHANNELS):
        super().__init__()
        self.board_size = board_size
        self.action_dim = board_size * board_size

        self.conv1 = nn.Conv2d(5, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)

        self.p_head = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.ReLU())
        self.p_fc = nn.Linear(2 * board_size * board_size, self.action_dim)

        self.v_head = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.ReLU())
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


def make_eval_fn(model: AZNet, device: str):
    """
    C++ MCTS expects eval_fn(state_tensor) -> (priors_1d_float32[n2], value_float)
    st is uint8 (5,N,N)
    """
    model.eval()

    @torch.no_grad()
    def eval_fn(st):
        x = torch.from_numpy(np.array(st, copy=False)).to(device, non_blocking=True)
        if x.ndim == 3:
            x = x.unsqueeze(0)

        logits, v = model(x)
        priors = (
            torch.softmax(logits, dim=-1)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        value = float(v.squeeze(0).detach().cpu().item())
        return priors, value

    return eval_fn


# ============================ MCTS POLICY ============================


def sample_action_from_policy(policy: np.ndarray, move_idx: int) -> Tuple[int, np.ndarray]:
    """
    C++ returns a normalized policy already.
    We still apply temperature early, then argmax later.
    """
    p = policy.astype(np.float32, copy=True)
    p_sum = float(p.sum())
    if p_sum <= 0:
        p[:] = 1.0 / len(p)
    else:
        p /= p_sum

    if move_idx < TEMPERATURE_MOVES:
        if TEMPERATURE <= 1e-8:
            a = int(np.argmax(p))
            return a, p
        logits = np.log(p + 1e-12) / TEMPERATURE
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum() + 1e-8
        a = int(np.random.choice(len(probs), p=probs))
        return a, probs

    a = int(np.argmax(p))
    return a, p


# ============================ TRAINING DATA ============================


@dataclass
class Sample:
    s: np.ndarray  # float32 (5,N,N)
    pi: np.ndarray  # float32 (n2,)
    z: float


# ============================ SELF-PLAY / EVAL (single game) ============================


def play_game_collect(new_model: AZNet, old_model: AZNet, size: int, new_starts: bool, seed_base: int, device: str) -> List[Sample]:
    game = az_engine.Game(size)
    n2 = game.n2()

    new_side = 1 if new_starts else 2

    # Per-game seeds so parallel games don't clone RNG streams
    mcts_new = az_engine.MCTS(
        cpuct=C_PUCT,
        n_sims=MCTS_SIMS,
        dir_alpha=DIRICHLET_ALPHA,
        dir_eps=DIRICHLET_EPS,
        seed=seed_base + 123,
    )
    mcts_old = az_engine.MCTS(
        cpuct=C_PUCT,
        n_sims=MCTS_SIMS,
        dir_alpha=DIRICHLET_ALPHA,
        dir_eps=DIRICHLET_EPS,
        seed=seed_base + 456,
    )

    eval_new = make_eval_fn(new_model, device)
    eval_old = make_eval_fn(old_model, device)

    traj: List[Tuple[np.ndarray, np.ndarray, int]] = []
    move_idx = 0
    safety_cap = size * size * 20

    while (not game.is_over()) and move_idx <= safety_cap:
        stm = game.current_player()  # 1 or 2

        if stm == new_side:
            policy, _best_move = mcts_new.run(game, eval_new)
        else:
            policy, _best_move = mcts_old.run(game, eval_old)

        policy = np.asarray(policy, dtype=np.float32)
        if policy.shape != (n2,):
            policy = policy.reshape(-1)[:n2]

        a, pi = sample_action_from_policy(policy, move_idx)

        s = np.array(game.state_tensor(), dtype=np.float32)  # (5,N,N)
        traj.append((s, pi.astype(np.float32), stm))

        r = game.apply(int(a))
        if r != 0:
            legal = game.legal_moves_all()
            if not legal:
                break
            ok = False
            for _ in range(8):
                a2 = int(np.random.choice(legal))
                if game.apply(a2) == 0:
                    ok = True
                    break
            if not ok:
                break

        move_idx += 1

    w = game.winner()  # 0 tie/none, 1/2 win
    out: List[Sample] = []
    for (s, pi_t, pl) in traj:
        z = 0.0 if w == 0 else (1.0 if w == pl else -1.0)
        out.append(Sample(s, pi_t, z))
    return out


@torch.no_grad()
def eval_game(new_model: AZNet, old_model: AZNet, size: int, new_starts: bool, seed_base: int, device: str) -> int:
    """
    Returns:
        +1 if new wins
        -1 if old wins
         0 if draw
    """
    game = az_engine.Game(size)
    n2 = game.n2()

    new_side = 1 if new_starts else 2

    mcts_new = az_engine.MCTS(cpuct=C_PUCT, n_sims=MCTS_SIMS, dir_alpha=0.0, dir_eps=0.0, seed=seed_base + 777)
    mcts_old = az_engine.MCTS(cpuct=C_PUCT, n_sims=MCTS_SIMS, dir_alpha=0.0, dir_eps=0.0, seed=seed_base + 999)

    eval_new_fn = make_eval_fn(new_model, device)
    eval_old_fn = make_eval_fn(old_model, device)

    safety_cap = size * size * 20
    move_idx = 0

    while (not game.is_over()) and move_idx <= safety_cap:
        stm = game.current_player()

        if stm == new_side:
            policy, _ = mcts_new.run(game, eval_new_fn)
        else:
            policy, _ = mcts_old.run(game, eval_old_fn)

        policy = np.asarray(policy, dtype=np.float32).reshape(-1)[:n2]

        if policy.sum() <= 0:
            legal = game.legal_moves_all()
            if not legal:
                break
            a = int(np.random.choice(legal))
        else:
            a = int(np.argmax(policy))

        r = game.apply(a)
        if r != 0:
            legal = game.legal_moves_all()
            if not legal:
                break
            ok = False
            for _ in range(8):
                a2 = int(np.random.choice(legal))
                if game.apply(a2) == 0:
                    ok = True
                    break
            if not ok:
                break

        move_idx += 1

    w = game.winner()
    if w == 0:
        return 0
    return +1 if w == new_side else -1


# ============================ TRAINING ============================

# ============================ SYMMETRY AUGMENTATION ============================

def _rotate_state(s: np.ndarray, k: int) -> np.ndarray:
    """
    Rotate state tensor by 90*k degrees.
    s: (5, N, N)
    """
    return np.rot90(s, k=k, axes=(1, 2))


def _rotate_policy(pi: np.ndarray, k: int, N: int) -> np.ndarray:
    """
    Rotate flattened policy by 90*k degrees.
    pi: (N*N,)
    """
    p = pi.reshape(N, N)
    p = np.rot90(p, k=k)
    return p.reshape(-1)


def _flip_state(s: np.ndarray) -> np.ndarray:
    """
    Horizontal mirror (left-right).
    """
    return np.flip(s, axis=2)


def _flip_policy(pi: np.ndarray, N: int) -> np.ndarray:
    """
    Horizontal mirror (left-right).
    """
    p = pi.reshape(N, N)
    p = np.flip(p, axis=1)
    return p.reshape(-1)


def augment_symmetries(sample: Sample) -> List[Sample]:
    """
    Apply D4 symmetry augmentation to a single sample.
    Returns 8 samples (4 rotations × mirror).
    """
    s, pi, z = sample.s, sample.pi, sample.z
    N = s.shape[1]

    out: List[Sample] = []

    for k in range(4):
        s_r = _rotate_state(s, k)
        pi_r = _rotate_policy(pi, k, N)
        out.append(Sample(s_r.copy(), pi_r.copy(), z))

        s_f = _flip_state(s_r)
        pi_f = _flip_policy(pi_r, N)
        out.append(Sample(s_f.copy(), pi_f.copy(), z))

    return out

def train_on_buffer(model: AZNet, buf: Deque[Sample]):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    data = list(buf)
    if not data:
        return

    for ep in tqdm(range(EPOCHS), desc="Train", leave=False):
        random.shuffle(data)
        losses = []
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
            s = torch.from_numpy(np.stack([b.s for b in batch])).to(DEVICE)
            pi = torch.from_numpy(np.stack([b.pi for b in batch])).to(DEVICE)
            z = torch.tensor([b.z for b in batch], dtype=torch.float32, device=DEVICE)

            logits, v = model(s)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(pi * logp).sum(dim=-1).mean()
            value_loss = F.mse_loss(v, z)
            loss = policy_loss + value_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(float(loss.item()))

        if losses:
            tqdm.write(f"  epoch {ep+1}/{EPOCHS} loss={sum(losses)/len(losses):.4f}")


def save_checkpoint(model: AZNet, path: str):
    torch.save({"state_dict": model.state_dict(), "board_size": model.board_size}, path)


def load_checkpoint(path: str):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = AZNet(ckpt["board_size"])
    m.load_state_dict(ckpt["state_dict"])
    return m


# ============================ MULTIPROCESSING WORKERS ============================

# Globals inside worker processes
_W_NEW = None
_W_OLD = None
_W_SIZE = None
_W_DEVICE = None


def _sd_to_cpu(sd):
    # Ensure state_dict tensors are CPU (picklable + cheap to share)
    return {k: v.detach().cpu() for k, v in sd.items()}


def worker_init(new_sd, old_sd, size: int, device: str, seed0: int):
    global _W_NEW, _W_OLD, _W_SIZE, _W_DEVICE
    _W_SIZE = size
    _W_DEVICE = device

    # Per-process RNG
    pid = os.getpid()
    set_seed(seed0 + pid)

    _W_NEW = AZNet(size)
    _W_NEW.load_state_dict(new_sd)
    _W_NEW.to(device)

    _W_OLD = AZNet(size)
    _W_OLD.load_state_dict(old_sd)
    _W_OLD.to(device)


def worker_selfplay(task):
    """
    task = (game_idx, new_starts, seed_base)
    returns List[Sample]
    """
    game_idx, new_starts, seed_base = task
    return play_game_collect(_W_NEW, _W_OLD, _W_SIZE, new_starts, seed_base, _W_DEVICE)


def worker_eval(task):
    """
    task = (game_idx, new_starts, seed_base)
    returns result int in {-1,0,+1}
    """
    game_idx, new_starts, seed_base = task
    return eval_game(_W_NEW, _W_OLD, _W_SIZE, new_starts, seed_base, _W_DEVICE)


def effective_workers(requested: int) -> int:
    # GPU/MPS cannot be safely shared across worker processes
    if DEVICE != "cpu":
        return 1
    return max(1, int(requested))


def run_selfplay_parallel(new_sd, old_sd, size: int, n_games: int, n_workers: int) -> List[Sample]:
    ctx = mp.get_context("spawn")
    tasks = [(i, (random.random() < 0.5), SEED + 100000 + i * 17) for i in range(n_games)]

    out: List[Sample] = []
    with ctx.Pool(
        processes=n_workers,
        initializer=worker_init,
        initargs=(new_sd, old_sd, size, DEVICE, SEED + 12345),
        maxtasksperchild=64,
    ) as pool:
        pbar = tqdm(total=n_games, desc="SelfPlay", leave=True)
        for samples in pool.imap_unordered(worker_selfplay, tasks, chunksize=1):
            out.extend(samples)
            pbar.update(1)
        pbar.close()
    return out


def run_eval_parallel(new_sd, old_sd, size: int, n_games: int, n_workers: int) -> Tuple[int, int, int]:
    ctx = mp.get_context("spawn")
    tasks = [(i, (random.random() < 0.5), SEED + 200000 + i * 19) for i in range(n_games)]

    nw = ow = dr = 0
    with ctx.Pool(
        processes=n_workers,
        initializer=worker_init,
        initargs=(new_sd, old_sd, size, DEVICE, SEED + 54321),
        maxtasksperchild=128,
    ) as pool:
        pbar = tqdm(total=n_games, desc="Eval", leave=False)
        for res in pool.imap_unordered(worker_eval, tasks, chunksize=1):
            if res == 1:
                nw += 1
            elif res == -1:
                ow += 1
            else:
                dr += 1
            pbar.update(1)
        pbar.close()

    return nw, ow, dr


# ============================ MAIN LOOP ============================


def main():
    size = BOARD_SIZE
    ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    best = load_checkpoint(ckpt_path)
    if best is None:
        best = AZNet(size)
        save_checkpoint(best, ckpt_path)

    best.to(DEVICE)

    # Worker count heuristic
    default_workers = (os.cpu_count() or 4) - 2
    default_workers = max(1, min(default_workers, 16))
    n_workers = effective_workers(default_workers)

    tqdm.write(f"DEVICE={DEVICE} | workers={n_workers} | cpu_count={os.cpu_count()}")

    replay: Deque[Sample] = deque(maxlen=REPLAY_BUFFER_SIZE)

    for it in range(1, N_ITERS + 1):
        cand = AZNet(size).to(DEVICE)
        cand.load_state_dict(best.state_dict())

        # --------- SELF-PLAY (parallel) ---------
        # self-play is cand vs best, but cand starts as best => both identical at this stage
        # (still fine; dirichlet + sampling provides diversity)
        cand_sd = _sd_to_cpu(cand.state_dict())
        best_sd = _sd_to_cpu(best.state_dict())

        all_samples = run_selfplay_parallel(
            new_sd=best_sd,
            old_sd=best_sd,
            size=size,
            n_games=N_SELFPLAY_GAMES,
            n_workers=n_workers,
        )


        # --------- ADD TO REPLAY WITH SYMMETRY AUGMENTATION ---------
        aug_count = 0
        for sample in all_samples:
            for aug in augment_symmetries(sample):
                replay.append(aug)
                aug_count += 1

        tqdm.write(f"[it {it:03d}] samples: {len(all_samples)} | replay: {len(replay)}")

        # --------- TRAIN (single process) ---------
        train_on_buffer(cand, replay)

        # --------- EVAL (parallel) ---------
        cand_sd2 = _sd_to_cpu(cand.state_dict())
        best_sd2 = _sd_to_cpu(best.state_dict())

        nw, ow, dr = run_eval_parallel(
            new_sd=cand_sd2,
            old_sd=best_sd2,
            size=size,
            n_games=EVAL_GAMES,
            n_workers=n_workers,
        )

        total = nw + ow + dr
        winrate = (nw + 0.5 * dr) / max(1, total)
        tqdm.write(f"[it {it:03d}] eval: new={nw} old={ow} draw={dr} | winrate={winrate:.3f}")

        if winrate >= ACCEPT_WINRATE:
            best = cand
            save_checkpoint(best, ckpt_path)
            tqdm.write(f"[it {it:03d}] ✅ accepted + checkpointed -> {ckpt_path}")
        else:
            tqdm.write(f"[it {it:03d}] ❌ rejected (kept previous best)")

    tqdm.write("Done.")


if __name__ == "__main__":
    main()