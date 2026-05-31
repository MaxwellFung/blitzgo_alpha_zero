#!/usr/bin/env python3
import argparse
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import az_engine
from move_ranker import load_scripted_model, ranked_moves


BOARD_SIZE = 7
NUM_STATES_SEARCHED = 10000000
RANKER_MODEL = "model/move_ranker_7x7.ts"
TOP_K = 12
WORKERS = os.cpu_count() or 1


class GameSession:
    def __init__(self, size: int, states: int, ranker_path: str, top_k: int, workers: int):
        self.size = size
        self.states = states
        self.ranker_path = ranker_path
        self.top_k = top_k
        self.workers = max(1, workers)
        self.lock = threading.Lock()
        if not Path(ranker_path).exists():
            raise FileNotFoundError(
                f"Missing 7x7 ranker model: {ranker_path}. "
                "Generate data with `python3 py/generate_training_data.py`, then train with "
                "`python3 py/train_move_ranker.py`."
            )
        self.model = load_scripted_model(ranker_path)
        self.mode = "engine"
        self.human_side = 1
        self.game = az_engine.Game(size)
        self.last_engine_info = None
        self.message = "New game."

    def reset(self, mode: str, human_side: int):
        with self.lock:
            self.mode = "pvp" if mode == "pvp" else "engine"
            self.human_side = 2 if human_side == 2 else 1
            self.game = az_engine.Game(self.size)
            self.last_engine_info = None
            self.message = "New game."
            return self.state_locked()

    def state(self):
        with self.lock:
            return self.state_locked()

    def state_locked(self):
        winner = self.game.winner() if self.game.is_over() else None
        return {
            "size": self.size,
            "mode": self.mode,
            "human_side": self.human_side,
            "current_player": self.game.current_player(),
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

            result = self.game.apply(move)
            if result != 0:
                self.message = "Illegal move."
                return self.state_locked()

            x = move % self.size + 1
            y = move // self.size + 1
            self.message = f"Player {3 - self.game.current_player()} played ({x}, {y})."
            self.last_engine_info = None
            return self.state_locked()

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

            search = az_engine.Minimax(max_states=self.states)
            candidates = ranked_moves(self.model, self.game)[: self.top_k]
            move = search.best_move_subset_parallel(self.game, candidates, self.workers)
            if move < 0:
                self.message = "Engine found no legal move."
                return self.state_locked()

            result = self.game.apply(move)
            if result != 0:
                self.message = "Engine selected an illegal move."
                return self.state_locked()

            self.last_engine_info = {
                "move": move,
                "x": move % self.size + 1,
                "y": move // self.size + 1,
                "kept": len(candidates),
                "states": search.states_searched(),
                "completed_depth": search.completed_depth(),
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
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE)
    parser.add_argument("--states", type=int, default=NUM_STATES_SEARCHED)
    parser.add_argument("--ranker", default=RANKER_MODEL)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--workers", type=int, default=WORKERS)
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
        max(1, args.workers),
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session, ui_dir))
    print(f"BlitzGo UI: http://{args.host}:{args.port}")
    print(
        f"Engine: top-{args.top_k}, states={args.states:,}, "
        f"workers={max(1, args.workers)}, ranker={args.ranker}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
