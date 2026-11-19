"""Zero-extra-dependency local HTTP server for the Three.js investment city.

Usage: uv run python play.py
       uv run python -m visualization.server --open --port 8765

The server binds to localhost by default and NEVER retrains the agent.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
from pathlib import Path
import threading
import webbrowser
from urllib.parse import urlsplit

from visualization.session import SimulationController

WEB_ROOT = Path(__file__).resolve().parent / "static"
LOGGER = logging.getLogger("industrial_visualization")


def make_handler(controller):
    class Handler(BaseHTTPRequestHandler):
        server_version = "IndustrialSimulator/1.0"

        def send_payload(self, status: int, body: bytes, mime: str):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, data):
            self.send_payload(status, json.dumps(data, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/state":
                return self.send_json(200, controller.snapshot())
            if path == "/api/health":
                return self.send_json(200, {"status": "ok", "model": "ppo_03", "source": "TradeSignalEnv"})
            if path == "/":
                path = "/index.html"
            if path not in ("/index.html", "/app.js", "/world.js", "/styles.css", "/favicon.svg"):
                return self.send_json(404, {"error": "Not found"})
            target = WEB_ROOT / path.lstrip("/")
            if not target.is_file():
                return self.send_json(404, {"error": "Asset not found"})
            mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if target.suffix in (".html", ".js", ".css", ".svg"):
                mime += "; charset=utf-8"
            self.send_payload(200, target.read_bytes(), mime)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/step", "/api/reset"):
                return self.send_json(404, {"error": "Unknown API endpoint"})
            try:
                raw_length = self.headers.get("Content-Length", "0")
                size = int(raw_length)
                if not 0 <= size <= 2048:
                    return self.send_json(413, {"error": "Request body too large"})
                payload = json.loads(self.rfile.read(size) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                if path == "/api/reset":
                    result = controller.reset(
                        seed=payload.get("seed", 60000),
                        split=payload.get("split", "holdout"),
                    )
                else:
                    result = controller.step()
                self.send_json(200, result)
            except (ValueError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception:
                LOGGER.exception("Simulator API failure")
                self.send_json(500, {"error": "Simulation error; check terminal output"})

        def log_message(self, fmt, *args):
            if not (self.path.startswith("/api/state") or self.path.startswith("/api/health")):
                LOGGER.debug("HTTP %s", fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Interactive Three.js + PPO 03 investment simulation")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1", help="Localhost only; do not expose to the internet")
    parser.add_argument("--seed", type=int, default=60000)
    parser.add_argument("--split", choices=("historical", "holdout"), default="holdout")
    parser.add_argument("--open", action="store_true", help="Open the URL in your default browser")
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("For safe local demonstration, bind only to 127.0.0.1 or localhost")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    controller = SimulationController(seed=args.seed, split=args.split)
    address = "127.0.0.1" if args.host == "localhost" else args.host
    try:
        with ThreadingHTTPServer((address, args.port), make_handler(controller)) as server:
            url = f"http://{address}:{server.server_port}/"
            print("\n" + "="*70, flush=True)
            print("  INDUSTRIAL INVESTMENT LAB  |  PPO 03  vs  RULE-BASED", flush=True)
            print(f"  Browser: {url}", flush=True)
            print(f"  Checkpoint: {controller.checkpoint}", flush=True)
            print("  24-month scenario. Every STEP emits verbose terminal actions.", flush=True)
            print("  Press Ctrl+C to stop. No retraining occurs.", flush=True)
            print("="*70 + "\n", flush=True)
            if args.open:
                threading.Timer(0.8, lambda: webbrowser.open(url)).start()
            server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nSimulation stopped.", flush=True)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
