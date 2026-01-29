# pip install torch numpy tqdm
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

import os, math, random, copy
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing as mp

# ================== ALPHAZERO-LIKE HYPERPARAMETERS ==================

BOARD_SIZE = 5

N_ITERS = 200
N_SELFPLAY_GAMES = 400
MCTS_SIMS = 200

C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
EPOCHS = 5

REPLAY_BUFFER_SIZE = 20000

TEMPERATURE_MOVES = 6
TEMPERATURE = 1.0

EVAL_GAMES = 50
ACCEPT_WINRATE = 0.55

SEED = 42
DEVICE = "cpu"  # "cuda" / "mps" if available

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = f"az_territory_size{BOARD_SIZE}.pt"

# ---- PARALLELIZATION / BATCHING KNOBS ----
PARALLEL_GAMES = 1
PREDICT_BATCH_MAX = 4096
MCTS_BATCH_SIMULATIONS = 1

USE_TF32 = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def resolve_device(dev: str) -> str:
    if dev == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if dev == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return dev


DEVICE = resolve_device(DEVICE)

if USE_TF32 and DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# ============================ GAME LOGIC ============================

def _piece_index(v: int) -> int:
    # map stone value in {-1,0,+1} -> {0,1,2}
    if v == -1:
        return 0
    if v == 0:
        return 1
    return 2

def _selfplay_worker(proc_id, cand_sd, best_sd, size, zob, n_games, out_q, seed):
    set_seed(seed + proc_id + 1)
    torch.set_num_threads(1)

    cand = AZNet(size)
    best = AZNet(size)
    cand.load_state_dict(cand_sd)
    best.load_state_dict(best_sd)

    samples = selfplay_parallel_collect(cand, best, size, zob, n_games)
    out_q.put(samples)

class Board:
    def __init__(self, size: int, zobrist_table: np.ndarray):
        self.size = size
        self.size_2 = size * size

        # CPU numpy arrays for fast Python-side logic
        self.stones = np.zeros((size, size), dtype=np.int8)
        self.territory = np.zeros((size, size), dtype=np.int8)

        self.territory_counts = {+1: 0, -1: 0}
        self.total_territory_count = 0

        self.move_history: List[Tuple[int, int]] = []

        # Zobrist hashing (stones-only) to replace expensive tuple hashing
        self._zob = zobrist_table  # shape: (size, size, 3) uint64
        self.stone_hash = np.uint64(0)

        # stores hashes of seen stone positions (stones-only), including initial
        self.move_set = set()
        self.move_set.add(int(self.stone_hash))

    def clone(self) -> "Board":
        b = Board.__new__(Board)
        b.size = self.size
        b.size_2 = self.size_2
        b.stones = self.stones.copy()
        b.territory = self.territory.copy()
        b.territory_counts = {+1: self.territory_counts[+1], -1: self.territory_counts[-1]}
        b.total_territory_count = self.total_territory_count
        b.move_history = self.move_history.copy()
        b._zob = self._zob
        b.stone_hash = np.uint64(self.stone_hash)
        b.move_set = set(self.move_set)
        return b

    def is_within_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size

    def isValidMove(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if not self.is_within_bounds(pos):
            return False
        return int(self.stones[y, x]) == 0

    def _set_stone(self, x: int, y: int, new_v: int):
        old_v = int(self.stones[y, x])
        if old_v == new_v:
            return
        self.stone_hash ^= self._zob[y, x, _piece_index(old_v)]
        self.stone_hash ^= self._zob[y, x, _piece_index(new_v)]
        self.stones[y, x] = np.int8(new_v)

    def would_be_duplicate(self, pos: Tuple[int, int], player: int) -> bool:
        x, y = pos
        if not self.isValidMove(pos):
            return True
        # prospective hash after placing player's stone at (x,y)
        h = np.uint64(self.stone_hash)
        h ^= self._zob[y, x, _piece_index(0)]
        h ^= self._zob[y, x, _piece_index(player)]
        return int(h) in self.move_set

    def isDuplicateMove(self) -> bool:
        h = int(self.stone_hash)
        if h in self.move_set:
            return True
        self.move_set.add(h)
        return False

    def incrementTerritory(self, player: int, delta: int):
        self.territory_counts[player] += delta
        self.total_territory_count += delta

    def totalTerritory(self) -> int:
        return self.total_territory_count

    def removeStone(self, pos: Tuple[int, int]):
        x, y = pos
        p = int(self.stones[y, x])
        if p != 0:
            self.incrementTerritory(p, -1)
        self._set_stone(x, y, 0)

    def remove_last_move(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        s = int(self.stones[y, x])
        t = int(self.territory[y, x])
        if s != 0 and t != 0 and s != t:
            self.incrementTerritory(t, +1)
            self.removeStone(pos)
            return True
        return False

    def removeSingleTerritory(self, pos: Tuple[int, int]):
        x, y = pos
        t = int(self.territory[y, x])
        s = int(self.stones[y, x])
        if t != 0 and s != 0 and t != s:
            self.incrementTerritory(t, -1)

    def generateValidMoves(self) -> List[Tuple[int, int]]:
        move_scores: Dict[Tuple[int, int], int] = {}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1), (1, 1), (-1, -1)]
        stones = self.stones
        n = self.size

        # score empties adjacent to stones
        ys, xs = np.nonzero(stones)
        for y, x in zip(ys.tolist(), xs.tolist()):
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and int(stones[ny, nx]) == 0:
                    move_scores[(nx, ny)] = move_scores.get((nx, ny), 0) + 1

        sorted_moves = sorted(move_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_moves = [m for (m, _) in sorted_moves[:20]]

        if not top_moves:
            empties = np.argwhere(stones == 0)
            top_moves = [(int(x), int(y)) for y, x in empties.tolist()]
        return top_moves

    def bfs_update_opponent_territory(self, player: int, start: Tuple[int, int]):
        q = deque([start])
        sx, sy = start
        if self.is_within_bounds((sx, sy)):
            tv = int(self.territory[sy, sx])
            if tv != player and tv != 0:
                self.incrementTerritory(tv, -1)
                self.territory[sy, sx] = 0

        while q:
            x, y = q.popleft()
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nx, ny = x + dx, y + dy
                if self.is_within_bounds((nx, ny)):
                    tv = int(self.territory[ny, nx])
                    if tv != player and tv != 0:
                        q.append((nx, ny))
                        self.incrementTerritory(tv, -1)
                        self.territory[ny, nx] = 0

    def bfs_enclosed_territory(self, player: int, start: Tuple[int, int], totalVisited: set):
        q = deque([start])
        totalVisited.add(start)
        walls_touched = set()
        visited = {start}
        n = self.size
        stones = self.stones

        while q:
            x, y = q.popleft()
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nx, ny = x + dx, y + dy
                boolTouchWall = False
                if ny < 0:
                    walls_touched.add("top")
                    boolTouchWall = True
                elif ny >= n:
                    walls_touched.add("bottom")
                    boolTouchWall = True
                if nx < 0:
                    walls_touched.add("left")
                    boolTouchWall = True
                elif nx >= n:
                    walls_touched.add("right")
                    boolTouchWall = True

                if len(walls_touched) > 2:
                    return None
                if boolTouchWall:
                    continue
                if (nx, ny) in visited:
                    continue

                if int(stones[ny, nx]) != player:
                    q.append((nx, ny))
                    totalVisited.add((nx, ny))
                    visited.add((nx, ny))

        return visited

    def update_territories(self, player: int, pos: Tuple[int, int]) -> set:
        x, y = pos
        total_territory = set()

        if int(self.territory[y, x]) == player:
            self.incrementTerritory(player, -1)
            self.territory[y, x] = 0
            return total_territory

        totalVisited = set()
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if self.is_within_bounds((nx, ny)) and int(self.stones[ny, nx]) != player and (nx, ny) not in totalVisited:
                enclosed = self.bfs_enclosed_territory(player, (nx, ny), totalVisited)
                if enclosed:
                    for (tx, ty) in enclosed:
                        cur = int(self.territory[ty, tx])
                        if cur != player:
                            if cur != 0:
                                self.incrementTerritory(cur, -1)
                            self.incrementTerritory(player, +1)
                            self.territory[ty, tx] = np.int8(player)
                        total_territory.add((tx, ty))
        return total_territory

    def remove_stones_in_territory(self, player: int, total_territory: set) -> bool:
        captured = False
        for (x, y) in total_territory:
            s = int(self.stones[y, x])
            if s != 0 and s != player:
                self.removeStone((x, y))
                captured = True
        return captured

    def update_other_stones(self, player: int, pos: Tuple[int, int]) -> bool:
        self.incrementTerritory(player, +1)
        total_territory = self.update_territories(player, pos)
        captured = self.remove_stones_in_territory(player, total_territory)
        if captured:
            self.bfs_update_opponent_territory(player, pos)
        if self.move_history:
            if self.remove_last_move(self.move_history[-1]):
                captured = True
        self.removeSingleTerritory(pos)
        return captured

    def placeStone(self, player: int, pos: Tuple[int, int]) -> int:
        x, y = pos
        if not self.isValidMove(pos):
            return 1

        # duplicate check uses zobrist prospective hash
        if self.would_be_duplicate(pos, player):
            return 2

        self._set_stone(x, y, player)

        if self.isDuplicateMove():
            self._set_stone(x, y, 0)
            return 2

        _ = self.update_other_stones(player, pos)
        self.move_history.append(pos)
        return 0

    def stability_test(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        player = int(self.territory[y, x])
        if player == 0:
            raise RuntimeError("Invalid stability test on empty territory.")
        countCorner = 0
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx, ny = x + dx, y + dy
            if self.is_within_bounds((nx, ny)):
                s = int(self.stones[ny, nx])
                if s != 0 and s != player:
                    countCorner += 1
        if countCorner >= 2:
            return False
        if countCorner == 1:
            if x == 0 or y == 0 or x == self.size - 1 or y == self.size - 1:
                return False
        return True

    def checkGameOver(self) -> bool:
        terr = self.territory
        n = self.size
        for y in range(n):
            for x in range(n):
                if int(terr[y, x]) != 0 and not self.stability_test((x, y)):
                    return False
        return True

    def winner(self) -> int:
        a = self.territory_counts[+1]
        b = self.territory_counts[-1]
        if a > b:
            return +1
        if b > a:
            return -1
        return 0


class Game:
    def __init__(self, size: int, zobrist_table: np.ndarray):
        self.board = Board(size, zobrist_table)
        self.turn = 0  # 0 => +1, 1 => -1
        self.moveCount = 0
        self.first_move = False

    def current_player(self) -> int:
        return +1 if self.turn == 0 else -1

    def switchPlayer(self):
        self.turn = 1 - self.turn
        self.moveCount += 1

    def placeStone(self, position: Tuple[int, int]) -> int:
        p = self.current_player()
        mv = self.board.placeStone(p, position)
        if mv == 0:
            self.switchPlayer()
        return mv

    def checkGameOver(self) -> bool:
        if self.board.totalTerritory() >= self.board.size_2:
            return self.board.checkGameOver()
        return False

    def simulateMove(self, position: Tuple[int, int]) -> Optional["Game"]:
        g = self.clone()
        if g.placeStone(position) == 0:
            return g
        return None

    def legal_actions(self) -> List[int]:
        moves = self.board.generateValidMoves()
        acts = []
        p = self.current_player()
        for (x, y) in moves:
            if self.board.isValidMove((x, y)) and not self.board.would_be_duplicate((x, y), p):
                acts.append(y * self.board.size + x)
        if not acts:
            n = self.board.size
            for y in range(n):
                for x in range(n):
                    if self.board.isValidMove((x, y)) and not self.board.would_be_duplicate((x, y), p):
                        acts.append(y * n + x)
        return acts

    def clone(self) -> "Game":
        g = Game.__new__(Game)
        g.board = self.board.clone()
        g.turn = self.turn
        g.moveCount = self.moveCount
        g.first_move = self.first_move
        return g


# ============================ NEURAL NET ============================

def encode_state(game: Game) -> torch.Tensor:
    b = game.board
    s = torch.zeros(5, b.size, b.size, dtype=torch.float32)
    stones = torch.from_numpy(b.stones.astype(np.int8))
    terr = torch.from_numpy(b.territory.astype(np.int8))

    s[0] = (stones == 1)
    s[1] = (stones == -1)
    s[2] = (terr == 1)
    s[3] = (terr == -1)
    s[4].fill_(1.0 if game.current_player() == 1 else 0.0)
    return s


class AZNet(nn.Module):
    def __init__(self, board_size: int, channels: int = 384):
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


@torch.no_grad()
def predict_batch(model: AZNet, games: List[Game]) -> Tuple[np.ndarray, np.ndarray]:
    if not games:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    model.eval()
    priors_list = []
    vals_list = []

    for i in range(0, len(games), PREDICT_BATCH_MAX):
        chunk = games[i : i + PREDICT_BATCH_MAX]
        xs = torch.stack([encode_state(g) for g in chunk]).to(DEVICE, non_blocking=True)
        logits, v = model(xs)
        p = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
        vv = v.detach().cpu().numpy().astype(np.float32)
        priors_list.append(p)
        vals_list.append(vv)

    priors = np.concatenate(priors_list, axis=0)
    vals = np.concatenate(vals_list, axis=0)
    return priors, vals


# ============================ MCTS (BATCHED) ============================

@dataclass
class EdgeStats:
    P: float
    N: int = 0
    W: float = 0.0

    @property
    def Q(self) -> float:
        return 0.0 if self.N == 0 else self.W / self.N


class MCTS:
    def __init__(self, model: AZNet):
        self.model = model
        self.tree: Dict[bytes, Dict[int, EdgeStats]] = {}

    def advance_to_child(self, parent_game: Game, action: int, child_game: Game):
        parent_key = self._key(parent_game)
        child_key = self._key(child_game)

        if parent_key not in self.tree:
            return
        if action not in self.tree[parent_key]:
            self.tree = {}
            return

        if child_key in self.tree:
            self.tree = {child_key: self.tree[child_key]}
        else:
            self.tree = {}

    def _key(self, game: Game) -> bytes:
        b = game.board
        # fast bytes key from contiguous ravel
        arr = np.concatenate([b.stones.ravel(), b.territory.ravel(), np.array([game.turn], dtype=np.int8)], axis=0)
        return arr.tobytes()

    def _add_dirichlet_noise(self, key: bytes):
        edges = self.tree.get(key, {})
        actions = list(edges.keys())
        if not actions:
            return
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(actions)).astype(np.float32)
        for i, a in enumerate(actions):
            edges[a].P = (1 - DIRICHLET_EPS) * edges[a].P + DIRICHLET_EPS * float(noise[i])

    def _expand_from_priors(self, game: Game, key: bytes, priors_row: np.ndarray):
        legal = game.legal_actions()
        edges: Dict[int, EdgeStats] = {}
        if legal:
            ps = np.array([priors_row[a] for a in legal], dtype=np.float32)
            ps = ps / (ps.sum() + 1e-8)
            for a, p in zip(legal, ps):
                edges[a] = EdgeStats(P=float(p))
        self.tree[key] = edges

    def run(self, root_game: Game, sims: int, add_noise: bool) -> np.ndarray:
        return self.run_batched([root_game], sims=sims, add_noise=add_noise)[0]

    def run_batched(self, root_games: List[Game], sims: int, add_noise: bool) -> List[np.ndarray]:
        keys = [self._key(g) for g in root_games]
        to_expand = []
        to_expand_keys = []

        for k, g in zip(keys, root_games):
            if k not in self.tree:
                to_expand.append(g)
                to_expand_keys.append(k)

        if to_expand:
            priors, _vals = predict_batch(self.model, to_expand)
            for g, k, p in zip(to_expand, to_expand_keys, priors):
                self._expand_from_priors(g, k, p)

        if add_noise:
            for k in keys:
                self._add_dirichlet_noise(k)

        for _ in range(sims):
            for _inner in range(MCTS_BATCH_SIMULATIONS):
                self._simulate_batch(root_games)

        out = []
        for g in root_games:
            k = self._key(g)
            v = np.zeros(g.board.size_2, dtype=np.float32)
            edges = self.tree.get(k, {})
            for a, st in edges.items():
                v[a] = st.N
            out.append(v)
        return out

    def _simulate_batch(self, root_games: List[Game]):
        leaf_games: List[Game] = []
        leaf_keys: List[bytes] = []
        leaf_paths: List[List[Tuple[bytes, int, int]]] = []
        terminal_backup: List[Tuple[List[Tuple[bytes, int, int]], int, float]] = []
        illegal_paths: List[List[Tuple[bytes, int, int]]] = []

        for rg in root_games:
            path: List[Tuple[bytes, int, int]] = []
            game = rg.clone()

            while True:
                key = self._key(game)

                if game.checkGameOver():
                    leaf_player = game.current_player()
                    w = game.board.winner()
                    z_leaf = 0.0 if w == 0 else (1.0 if w == leaf_player else -1.0)
                    terminal_backup.append((path, leaf_player, z_leaf))
                    break

                if key not in self.tree:
                    leaf_games.append(game)
                    leaf_keys.append(key)
                    leaf_paths.append(path)
                    break

                edges = self.tree[key]
                if not edges:
                    leaf_player = game.current_player()
                    w = game.board.winner()
                    z_leaf = 0.0 if w == 0 else (1.0 if w == leaf_player else -1.0)
                    terminal_backup.append((path, leaf_player, z_leaf))
                    break

                total_N = 0
                for st in edges.values():
                    total_N += st.N

                best_a, best_u = None, -1e9
                for a, st in edges.items():
                    u = st.Q + C_PUCT * st.P * math.sqrt(total_N + 1e-8) / (1 + st.N)
                    if u > best_u:
                        best_u, best_a = u, a

                if best_a is None:
                    leaf_player = game.current_player()
                    w = game.board.winner()
                    z_leaf = 0.0 if w == 0 else (1.0 if w == leaf_player else -1.0)
                    terminal_backup.append((path, leaf_player, z_leaf))
                    break

                pl = game.current_player()
                path.append((key, best_a, pl))

                x = best_a % game.board.size
                y = best_a // game.board.size
                nxt = game.simulateMove((x, y))
                if nxt is None:
                    illegal_paths.append(path)
                    break
                game = nxt

        # punish illegal traversals
        for path in illegal_paths:
            for (k, a, _pl) in path:
                if k in self.tree and a in self.tree[k]:
                    self.tree[k][a].N += 1
                    self.tree[k][a].W += -1.0

        # terminal backups (no NN)
        for path, leaf_player, z_leaf in terminal_backup:
            for (k, a, pl) in path:
                val = z_leaf if pl == leaf_player else -z_leaf
                self.tree[k][a].N += 1
                self.tree[k][a].W += float(val)

        # NN eval for leaves
        if leaf_games:
            priors, vals = predict_batch(self.model, leaf_games)
            for game, key, path, p, v in zip(leaf_games, leaf_keys, leaf_paths, priors, vals):
                if key not in self.tree:
                    self._expand_from_priors(game, key, p)
                leaf_player = game.current_player()
                z_leaf = float(v)
                for (k, a, pl) in path:
                    val = z_leaf if pl == leaf_player else -z_leaf
                    self.tree[k][a].N += 1
                    self.tree[k][a].W += float(val)


def sample_action(visits: np.ndarray, move_idx: int) -> Tuple[int, np.ndarray]:
    v = visits.copy().astype(np.float32)
    if v.sum() <= 0:
        a = int(np.random.randint(0, len(v)))
        pi = np.ones_like(v) / len(v)
        return a, pi
    if move_idx < TEMPERATURE_MOVES:
        t = TEMPERATURE
        probs = np.power(v, 1.0 / max(t, 1e-8))
        probs = probs / (probs.sum() + 1e-8)
        a = int(np.random.choice(len(v), p=probs))
        return a, probs
    a = int(np.argmax(v))
    pi = v / (v.sum() + 1e-8)
    return a, pi


# ============================ TRAINING ============================

@dataclass
class Sample:
    s: torch.Tensor
    pi: torch.Tensor
    z: float


def make_zobrist(size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    # (y,x,pieceIndex) uint64
    return rng.randint(0, np.iinfo(np.uint64).max, size=(size, size, 3), dtype=np.uint64)


def selfplay_parallel_collect(cand: AZNet, best: AZNet, size: int, zob: np.ndarray, n_games: int) -> List[Sample]:
    samples_out: List[Sample] = []
    done = 0

    mcts_cand = MCTS(cand)
    mcts_best = MCTS(best)

    pbar = tqdm(total=n_games, desc="SelfPlay", leave=False)

    while done < n_games:
        batch_n = min(PARALLEL_GAMES, n_games - done)
        games = [Game(size, zob) for _ in range(batch_n)]
        new_sides = [+1 if random.random() < 0.5 else -1 for _ in range(batch_n)]
        traj: List[List[Tuple[torch.Tensor, torch.Tensor, int]]] = [[] for _ in range(batch_n)]
        move_idx = 0

        active = [True] * batch_n
        safety_cap = size * size * 20

        while any(active) and move_idx <= safety_cap:
            idx_cand = []
            idx_best = []

            for i, g in enumerate(games):
                if not active[i]:
                    continue
                if g.checkGameOver():
                    active[i] = False
                    continue
                stm = g.current_player()
                model_is_cand = (stm == new_sides[i])
                if model_is_cand:
                    idx_cand.append(i)
                else:
                    idx_best.append(i)

            if not idx_cand and not idx_best:
                break

            if idx_cand:
                roots = [games[i] for i in idx_cand]
                visits_list = mcts_cand.run_batched(roots, sims=MCTS_SIMS, add_noise=True)
                for i, visits in zip(idx_cand, visits_list):
                    a, pi = sample_action(visits, move_idx)
                    traj[i].append((encode_state(games[i]), torch.from_numpy(pi).float(), games[i].current_player()))
                    x, y = a % size, a // size
                    parent = games[i]
                    nxt = parent.simulateMove((x, y))
                    if nxt is None:
                        legal = games[i].legal_actions()
                        if not legal:
                            active[i] = False
                        else:
                            a2 = int(np.random.choice(legal))
                            pi2 = np.zeros(size * size, dtype=np.float32)
                            pi2[a2] = 1.0
                            traj[i].append((encode_state(games[i]), torch.from_numpy(pi2).float(), games[i].current_player()))
                            x2, y2 = a2 % size, a2 // size
                            nxt2 = games[i].simulateMove((x2, y2))
                            if nxt2 is None:
                                active[i] = False
                            else:
                                games[i] = nxt2
                    else:
                        games[i] = nxt
                        mcts_cand.advance_to_child(parent, a, nxt)

            if idx_best:
                roots = [games[i] for i in idx_best]
                visits_list = mcts_best.run_batched(roots, sims=MCTS_SIMS, add_noise=True)
                for i, visits in zip(idx_best, visits_list):
                    a, pi = sample_action(visits, move_idx)
                    traj[i].append((encode_state(games[i]), torch.from_numpy(pi).float(), games[i].current_player()))
                    x, y = a % size, a // size
                    parent = games[i]
                    nxt = parent.simulateMove((x, y))
                    if nxt is None:
                        legal = games[i].legal_actions()
                        if not legal:
                            active[i] = False
                        else:
                            a2 = int(np.random.choice(legal))
                            pi2 = np.zeros(size * size, dtype=np.float32)
                            pi2[a2] = 1.0
                            traj[i].append((encode_state(games[i]), torch.from_numpy(pi2).float(), games[i].current_player()))
                            x2, y2 = a2 % size, a2 // size
                            nxt2 = games[i].simulateMove((x2, y2))
                            if nxt2 is None:
                                active[i] = False
                            else:
                                games[i] = nxt2
                    else:
                        games[i] = nxt
                        mcts_best.advance_to_child(parent, a, nxt)

            move_idx += 1

            for i, g in enumerate(games):
                if active[i] and g.checkGameOver():
                    active[i] = False

        for i, g in enumerate(games):
            w = g.board.winner()
            for (s, pi_t, pl) in traj[i]:
                z = 0.0 if w == 0 else (1.0 if w == pl else -1.0)
                samples_out.append(Sample(s=s, pi=pi_t, z=z))

        done += batch_n
        pbar.update(batch_n)

    pbar.close()
    return samples_out


@torch.no_grad()
def eval_models_parallel(new_model: AZNet, old_model: AZNet, size: int, zob: np.ndarray) -> Tuple[int, int, int]:
    new_model.eval()
    old_model.eval()

    nw = ow = dr = 0
    done = 0

    mcts_new = MCTS(new_model)
    mcts_old = MCTS(old_model)

    pbar = tqdm(total=EVAL_GAMES, desc="Eval", leave=False)

    while done < EVAL_GAMES:
        batch_n = min(PARALLEL_GAMES, EVAL_GAMES - done)
        games = [Game(size, zob) for _ in range(batch_n)]
        new_sides = [+1 if random.random() < 0.5 else -1 for _ in range(batch_n)]
        active = [True] * batch_n
        safety_cap = size * size * 20
        move_idx = 0

        while any(active) and move_idx <= safety_cap:
            idx_new = []
            idx_old = []

            for i, g in enumerate(games):
                if not active[i]:
                    continue
                if g.checkGameOver():
                    active[i] = False
                    continue
                stm = g.current_player()
                model_is_new = (stm == new_sides[i])
                if model_is_new:
                    idx_new.append(i)
                else:
                    idx_old.append(i)

            if not idx_new and not idx_old:
                break

            if idx_new:
                roots = [games[i] for i in idx_new]
                visits_list = mcts_new.run_batched(roots, sims=MCTS_SIMS, add_noise=False)
                for i, visits in zip(idx_new, visits_list):
                    a = int(np.argmax(visits)) if visits.sum() > 0 else int(np.random.randint(0, size * size))
                    x, y = a % size, a // size
                    parent = games[i]
                    nxt = parent.simulateMove((x, y))
                    if nxt is None:
                        legal = games[i].legal_actions()
                        if not legal:
                            active[i] = False
                        else:
                            a2 = int(np.random.choice(legal))
                            x2, y2 = a2 % size, a2 // size
                            nxt2 = games[i].simulateMove((x2, y2))
                            if nxt2 is None:
                                active[i] = False
                            else:
                                games[i] = nxt2
                    else:
                        games[i] = nxt
                        mcts_new.advance_to_child(parent, a, nxt)

            if idx_old:
                roots = [games[i] for i in idx_old]
                visits_list = mcts_old.run_batched(roots, sims=MCTS_SIMS, add_noise=False)
                for i, visits in zip(idx_old, visits_list):
                    a = int(np.argmax(visits)) if visits.sum() > 0 else int(np.random.randint(0, size * size))
                    x, y = a % size, a // size
                    parent = games[i]
                    nxt = parent.simulateMove((x, y))
                    if nxt is None:
                        legal = games[i].legal_actions()
                        if not legal:
                            active[i] = False
                        else:
                            a2 = int(np.random.choice(legal))
                            x2, y2 = a2 % size, a2 // size
                            nxt2 = games[i].simulateMove((x2, y2))
                            if nxt2 is None:
                                active[i] = False
                            else:
                                games[i] = nxt2
                    else:
                        games[i] = nxt
                        mcts_old.advance_to_child(parent, a, nxt)

            move_idx += 1
            for i, g in enumerate(games):
                if active[i] and g.checkGameOver():
                    active[i] = False

        for i, g in enumerate(games):
            w = g.board.winner()
            if w == 0:
                dr += 1
            elif w == new_sides[i]:
                nw += 1
            else:
                ow += 1

        done += batch_n
        pbar.update(batch_n)

    pbar.close()
    return nw, ow, dr


def train_on_buffer(model: AZNet, buf: deque):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    data = list(buf)
    for ep in tqdm(range(EPOCHS), desc="Train", leave=False):
        random.shuffle(data)
        losses = []
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
            s = torch.stack([b.s for b in batch]).to(DEVICE, non_blocking=True)
            pi = torch.stack([b.pi for b in batch]).to(DEVICE, non_blocking=True)
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


def load_checkpoint(path: str) -> Optional[AZNet]:
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu")
    m = AZNet(ckpt["board_size"])
    m.load_state_dict(ckpt["state_dict"])
    return m


# =============================== MAIN ===============================

def main():
    size = BOARD_SIZE
    ckpt_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    # deterministic zobrist per board size + seed
    zob = make_zobrist(size, seed=SEED + 12345)

    best = load_checkpoint(ckpt_path)
    if best is None:
        best = AZNet(size)
        save_checkpoint(best, ckpt_path)

    best.to(DEVICE)

    replay = deque(maxlen=REPLAY_BUFFER_SIZE)

    for it in range(1, N_ITERS + 1):
        cand = AZNet(size).to(DEVICE)
        cand.load_state_dict(copy.deepcopy(best.state_dict()))

        # ---- multiprocessing self-play ----
        n_workers = max(1, (os.cpu_count() or 1) - 1)
        games_per_worker = N_SELFPLAY_GAMES // n_workers
        extras = N_SELFPLAY_GAMES % n_workers

        cand_sd = {k: v.cpu() for k, v in cand.state_dict().items()}
        best_sd = {k: v.cpu() for k, v in best.state_dict().items()}

        out_q = mp.Queue()
        procs = []

        for i in range(n_workers):
            ng = games_per_worker + (1 if i < extras else 0)
            p = mp.Process(
                target=_selfplay_worker,
                args=(i, cand_sd, best_sd, size, zob, ng, out_q, SEED),
            )
            p.start()
            procs.append(p)

        all_samples = []
        for _ in range(n_workers):
            all_samples.extend(out_q.get())

        for p in procs:
            p.join()
        for s in all_samples:
            replay.append(s)

        tqdm.write(f"[it {it:03d}] selfplay samples added: {len(all_samples)} | replay size: {len(replay)}")

        train_on_buffer(cand, replay)

        nw, ow, dr = eval_models_parallel(cand, best, size, zob)
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
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()