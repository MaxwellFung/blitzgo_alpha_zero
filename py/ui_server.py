#!/usr/bin/env python3
import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import az_engine
from training.move_ranker import encode_game, load_scripted_model, ranked_move_predictions


BOARD_SIZE = 7
NUM_STATES_SEARCHED = 2500000
RANKER_MODEL = "model/move_ranker.ts"
VALUE_MODEL = "model/value_ranker.bin"
TOP_K = 12
INTERNAL_TOP_K = 0
WORKERS = os.cpu_count() or 1
GAME_DATA_DIR = "data/ui_games"


class GameSession:
    def __init__(
        self,
        size: int,
        states: int,
        ranker_path: str,
        top_k: int,
        internal_top_k: int,
        value_model: str,
        workers: int,
        game_data_dir: str = GAME_DATA_DIR,
    ):
        self.size = size
        self.states = states
        self.ranker_path = ranker_path
        self.top_k = top_k
        self.internal_top_k = max(0, internal_top_k)
        self.value_model = value_model
        self.workers = max(1, workers)
        self.game_data_dir = Path(game_data_dir) if game_data_dir else None
        self.lock = threading.Lock()
        if not Path(ranker_path).exists():
            raise FileNotFoundError(
                f"Missing 7x7 ranker model: {ranker_path}. "
                "Generate data with `python3 py/generate_training_data.py`, then train with "
                "`python3 training/train_move_ranker.py`."
            )
        if value_model and not Path(value_model).exists():
            raise FileNotFoundError(f"Missing native value model: {value_model}")
        self.model = load_scripted_model(ranker_path)
        self.value_evaluator = az_engine.Minimax(value_model=value_model)
        self.heuristic_evaluator = az_engine.Minimax()
        self.mode = "engine"
        self.human_side = 1
        self.game = az_engine.Game(size)
        self.game_id = ""
        self.game_positions = []
        self.last_engine_info = None
        self.message = "New game."
        self.start_recording_locked()

    def start_recording_locked(self):
        self.game_id = f"{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        self.game_positions = []

    def record_move_locked(self, move: int, actor: str):
        player = self.game.current_player()
        before_p1 = self.game.score_p1()
        before_p2 = self.game.score_p2()
        return {
            "board": encode_game(self.game),
            "move": move,
            "player": player,
            "actor": actor,
            "score_p1_before": before_p1,
            "score_p2_before": before_p2,
            "value_evaluation": int(self.value_evaluator.evaluate(self.game, player)),
            "heuristic_evaluation": int(self.heuristic_evaluator.evaluate(self.game, player)),
        }

    def save_game_locked(self):
        if self.game_data_dir is None or not self.game_positions:
            return
        self.game_data_dir.mkdir(parents=True, exist_ok=True)
        completed = self.game.is_over()
        final_p1 = self.game.score_p1()
        final_p2 = self.game.score_p2()
        winner = self.game.winner() if completed else 0
        outcome_targets = np.full(len(self.game_positions), np.nan, dtype=np.float32)
        if completed:
            for index, position in enumerate(self.game_positions):
                difference = final_p1 - final_p2
                if position["player"] == 2:
                    difference = -difference
                outcome_targets[index] = np.tanh(difference / 5.0)
        path = self.game_data_dir / f"game_{self.game_id}.npz"
        np.savez_compressed(
            path,
            boards=np.stack([item["board"] for item in self.game_positions]),
            moves=np.asarray([item["move"] for item in self.game_positions], dtype=np.int16),
            players=np.asarray([item["player"] for item in self.game_positions], dtype=np.int8),
            actors=np.asarray([item["actor"] for item in self.game_positions]),
            score_p1_before=np.asarray(
                [item["score_p1_before"] for item in self.game_positions],
                dtype=np.int16,
            ),
            score_p2_before=np.asarray(
                [item["score_p2_before"] for item in self.game_positions],
                dtype=np.int16,
            ),
            score_p1_after=np.asarray(
                [item["score_p1_after"] for item in self.game_positions],
                dtype=np.int16,
            ),
            score_p2_after=np.asarray(
                [item["score_p2_after"] for item in self.game_positions],
                dtype=np.int16,
            ),
            value_evaluations=np.asarray(
                [item["value_evaluation"] for item in self.game_positions],
                dtype=np.int16,
            ),
            heuristic_evaluations=np.asarray(
                [item["heuristic_evaluation"] for item in self.game_positions],
                dtype=np.int16,
            ),
            outcome_targets=outcome_targets,
            completed=np.asarray([completed], dtype=np.bool_),
            winner=np.asarray([winner], dtype=np.int8),
            final_score_p1=np.asarray([final_p1], dtype=np.int16),
            final_score_p2=np.asarray([final_p2], dtype=np.int16),
            mode=np.asarray([self.mode]),
            human_side=np.asarray([self.human_side], dtype=np.int8),
        )

    def finish_recorded_move_locked(self, position: dict):
        position["score_p1_after"] = self.game.score_p1()
        position["score_p2_after"] = self.game.score_p2()
        self.game_positions.append(position)
        self.save_game_locked()

    def reset(self, mode: str, human_side: int):
        with self.lock:
            self.save_game_locked()
            self.mode = "pvp" if mode == "pvp" else "engine"
            self.human_side = 2 if human_side == 2 else 1
            self.game = az_engine.Game(self.size)
            self.start_recording_locked()
            self.last_engine_info = None
            self.message = "New game."
            return self.state_locked()

    def state(self):
        with self.lock:
            return self.state_locked()

    def archive(self):
        with self.lock:
            self.save_game_locked()
            if self.game_data_dir is None or not self.game_data_dir.exists():
                return {"games": [], "current_game_id": self.game_id if self.game_positions else None}
            games = []
            for path in sorted(self.game_data_dir.glob("game_*.npz"), reverse=True):
                try:
                    with np.load(path) as game:
                        moves = int(game["moves"].shape[0])
                        games.append(
                            {
                                "id": path.stem.removeprefix("game_"),
                                "moves": moves,
                                "completed": bool(game["completed"][0]),
                                "winner": int(game["winner"][0]),
                                "score_p1": int(game["final_score_p1"][0]),
                                "score_p2": int(game["final_score_p2"][0]),
                                "mode": str(game["mode"][0]),
                            }
                        )
                except (KeyError, OSError, ValueError):
                    continue
            return {
                "games": games,
                "current_game_id": self.game_id if self.game_positions else None,
            }

    def archived_game(self, game_id: str):
        with self.lock:
            if self.game_data_dir is None:
                raise FileNotFoundError("Game recording is disabled.")
            if not game_id or any(character not in "0123456789abcdef_" for character in game_id):
                raise ValueError("Invalid archived game id.")
            path = self.game_data_dir / f"game_{game_id}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing archived game: {game_id}")
            with np.load(path) as game:
                boards = game["boards"]
                moves = game["moves"]
                players = game["players"]
                actors = game["actors"]
                score_p1_after = game["score_p1_after"]
                score_p2_after = game["score_p2_after"]
                value_evaluations = game["value_evaluations"]
                heuristic_evaluations = game["heuristic_evaluations"]
                frames = [
                    {
                        "ply": 0,
                        "stones": boards[0, 0].astype(np.uint8).reshape(-1).tolist(),
                        "territories": np.where(
                            boards[0, 2],
                            1,
                            np.where(boards[0, 3], 2, 0),
                        ).astype(np.uint8).reshape(-1).tolist(),
                        "current_player": int(players[0]),
                        "move": None,
                        "actor": None,
                        "score_p1": int(game["score_p1_before"][0]),
                        "score_p2": int(game["score_p2_before"][0]),
                        "value_evaluation": int(value_evaluations[0]),
                        "heuristic_evaluation": int(heuristic_evaluations[0]),
                    }
                ]
                replay = az_engine.Game(self.size)
                for index, move in enumerate(moves):
                    move = int(move)
                    if replay.apply(move) != 0:
                        raise RuntimeError(f"Archived game contains illegal move {move}.")
                    current_player = replay.current_player()
                    frames.append(
                        {
                            "ply": index + 1,
                            "stones": list(replay.stones()),
                            "territories": list(replay.territories()),
                            "current_player": current_player,
                            "move": move,
                            "actor": str(actors[index]),
                            "score_p1": int(score_p1_after[index]),
                            "score_p2": int(score_p2_after[index]),
                            "value_evaluation": int(
                                self.value_evaluator.evaluate(replay, current_player)
                            ),
                            "heuristic_evaluation": int(
                                self.heuristic_evaluator.evaluate(replay, current_player)
                            ),
                        }
                    )
                return {
                    "id": game_id,
                    "completed": bool(game["completed"][0]),
                    "winner": int(game["winner"][0]),
                    "score_p1": int(game["final_score_p1"][0]),
                    "score_p2": int(game["final_score_p2"][0]),
                    "frames": frames,
                }

    def state_locked(self):
        winner = self.game.winner() if self.game.is_over() else None
        current_player = self.game.current_player()
        return {
            "size": self.size,
            "mode": self.mode,
            "human_side": self.human_side,
            "current_player": current_player,
            "value_evaluation": int(self.value_evaluator.evaluate(self.game, current_player)),
            "heuristic_evaluation": int(
                self.heuristic_evaluator.evaluate(self.game, current_player)
            ),
            "value_model": self.value_model or None,
            "is_over": self.game.is_over(),
            "winner": winner,
            "score_p1": self.game.score_p1(),
            "score_p2": self.game.score_p2(),
            "stones": list(self.game.stones()),
            "territories": list(self.game.territories()),
            "last_engine_info": self.last_engine_info,
            "message": self.message,
        }

    def apply_human_move(self, move: int):
        with self.lock:
            if self.game.is_over():
                self.message = "Game is already over."
                return self.state_locked()

            if self.mode == "engine" and self.game.current_player() != self.human_side:
                self.message = "It is the engine's turn."
                return self.state_locked()

            position = self.record_move_locked(move, "human")
            result = self.game.apply(move)
            if result != 0:
                self.message = "Illegal move."
                return self.state_locked()
            self.finish_recorded_move_locked(position)

            x = move % self.size + 1
            y = move // self.size + 1
            self.message = f"Player {3 - self.game.current_player()} played ({x}, {y})."
            self.last_engine_info = None
            return self.state_locked()

    def cnn_top_k_locked(self):
        predictions = ranked_move_predictions(self.model, self.game, self.top_k)
        stones = list(self.game.stones())
        territories = list(self.game.territories())
        current_player = self.game.current_player()
        opponent = 2 if current_player == 1 else 1
        enriched = []
        for rank, item in enumerate(predictions, start=1):
            move = int(item["move"])
            result = self.game.apply(move)
            legal = result == 0
            if legal:
                self.game.undo()

            territory_owner = int(territories[move])
            if stones[move]:
                reason = "occupied"
            elif not territory_owner:
                reason = "empty"
            elif territory_owner == current_player:
                reason = "own territory"
            elif territory_owner == opponent:
                reason = "opponent territory"
            else:
                reason = f"territory P{territory_owner}"

            enriched.append(
                {
                    "rank": rank,
                    "move": move,
                    "x": move % self.size + 1,
                    "y": move // self.size + 1,
                    "probability": item["probability"],
                    "logit": item["logit"],
                    "legal": legal,
                    "apply_result": result,
                    "territory_owner": territory_owner,
                    "reason": reason,
                }
            )
        return enriched

    def cnn_top_k(self):
        with self.lock:
            return self.cnn_top_k_locked()

    def minimax_rows(self, search_info: dict, cnn_top_k: list[dict]) -> list[dict]:
        cnn_by_move = {item["move"]: item for item in cnn_top_k}
        best_move = int(search_info["best_move"])
        rows = []
        for rank, (move, score, searched_states) in enumerate(
            zip(search_info["moves"], search_info["scores"], search_info["states"]),
            start=1,
        ):
            move = int(move)
            cnn = cnn_by_move.get(move, {})
            rows.append(
                {
                    "rank": rank,
                    "move": move,
                    "x": move % self.size + 1,
                    "y": move // self.size + 1,
                    "score": int(score),
                    "states": int(searched_states),
                    "chosen": move == best_move,
                    "cnn_rank": cnn.get("rank"),
                    "cnn_probability": cnn.get("probability"),
                }
            )
        return rows

    def apply_engine_move(self):
        with self.lock:
            if self.mode != "engine":
                self.message = "Engine is disabled in Human vs Human mode."
                return self.state_locked()
            if self.game.is_over():
                self.message = "Game is already over."
                return self.state_locked()
            if self.game.current_player() == self.human_side:
                self.message = "It is the human's turn."
                return self.state_locked()

            search = az_engine.Minimax(
                max_states=self.states,
                internal_top_k=self.internal_top_k,
                value_model=self.value_model,
            )
            cnn_top_k = self.cnn_top_k_locked()
            candidates = [item["move"] for item in cnn_top_k]
            search_info = search.best_move_subset_parallel_info(
                self.game,
                candidates,
                self.workers,
            )
            move = int(search_info["best_move"])
            minimax_ranked = self.minimax_rows(search_info, cnn_top_k)
            if move < 0:
                self.message = "Engine found no legal move."
                self.last_engine_info = {
                    "cnn_top_k": cnn_top_k,
                    "minimax_ranked": minimax_ranked,
                }
                return self.state_locked()

            position = self.record_move_locked(move, "engine")
            result = self.game.apply(move)
            if result != 0:
                self.message = "Engine selected an illegal move."
                return self.state_locked()
            self.finish_recorded_move_locked(position)

            self.last_engine_info = {
                "move": move,
                "x": move % self.size + 1,
                "y": move // self.size + 1,
                "kept": len(candidates),
                "internal_top_k": self.internal_top_k,
                "value_model": self.value_model or None,
                "states": int(search_info["states_searched"]),
                "completed_depth": int(search_info["completed_depth"]),
                "cnn_top_k": cnn_top_k,
                "minimax_ranked": minimax_ranked,
            }
            info = self.last_engine_info
            self.message = (
                f"Engine played ({info['x']}, {info['y']}); "
                f"searched {info['states']:,} states, depth {info['completed_depth']}."
            )
            return self.state_locked()


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def write_json(handler, data, status=200):
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(session: GameSession, ui_dir: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ui_dir), **kwargs)

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/api/state":
                    write_json(self, session.state())
                    return
                if path == "/api/archive":
                    write_json(self, session.archive())
                    return
                if path.startswith("/api/archive/"):
                    write_json(self, session.archived_game(path.removeprefix("/api/archive/")))
                    return
                if path == "/":
                    self.path = "/index.html"
                super().do_GET()
            except Exception as exc:
                write_json(self, {"error": str(exc)}, status=500)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                payload = read_json(self)
                if path == "/api/reset":
                    data = session.reset(
                        payload.get("mode", "engine"),
                        int(payload.get("human_side", 1)),
                    )
                elif path == "/api/move":
                    data = session.apply_human_move(int(payload["move"]))
                elif path == "/api/cnn_top_k":
                    data = {"cnn_top_k": session.cnn_top_k()}
                elif path == "/api/engine":
                    data = session.apply_engine_move()
                else:
                    write_json(self, {"error": "Not found"}, status=404)
                    return
                write_json(self, data)
            except Exception as exc:
                write_json(self, {"error": str(exc)}, status=500)

        def log_message(self, format, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Run the BlitzGo browser UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE)
    parser.add_argument("--states", type=int, default=NUM_STATES_SEARCHED)
    parser.add_argument("--ranker", default=RANKER_MODEL)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--internal-top-k",
        type=int,
        default=INTERNAL_TOP_K,
        help="Experimental minimax pruning below the root. 0 searches all internal moves.",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--value-model", default=VALUE_MODEL)
    parser.add_argument("--game-data-dir", default=GAME_DATA_DIR)
    args = parser.parse_args()
    if args.board_size != BOARD_SIZE:
        raise SystemExit(
            f"The CNN ranker currently supports only {BOARD_SIZE}x{BOARD_SIZE}; "
            f"got --board-size {args.board_size}."
        )

    root = Path(__file__).resolve().parent.parent
    ui_dir = root / "ui"
    session = GameSession(
        args.board_size,
        args.states,
        args.ranker,
        args.top_k,
        args.internal_top_k,
        args.value_model,
        max(1, args.workers),
        args.game_data_dir,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session, ui_dir))
    print(f"BlitzGo UI: http://{args.host}:{args.port}")
    print(
        f"Engine: top-{args.top_k}, internal_top_k={max(0, args.internal_top_k)}, "
        f"states={args.states:,}, "
        f"workers={max(1, args.workers)}, ranker={args.ranker}, "
        f"value_model={args.value_model or 'heuristic'}, "
        f"game_data_dir={args.game_data_dir or 'disabled'}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
