#!/usr/bin/env python3
"""
web_ui.py - a browser version of the FtoCNeuralNet lesson.

Run it, and it opens a page in your web browser where you can watch the
neuron train with real charts:

    python3 web_ui.py

How it works:
  - This file is a tiny web server (Python's built-in http.server).
  - When you press "Train" on the page, the browser asks this server to
    train a fresh neuron. The training uses exactly the same code as the
    command-line version: the Neuron class and train() loop in neuron.py.
  - The server records w, b and the loss after every epoch and sends that
    history back as JSON. The page (web_ui.html) then plays it back like
    a movie, so you can pause, rewind and scrub through training.

So all of the learning happens in Python. The browser only draws.
"""

import argparse
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from neuron import (
    INPUT_SCALE,
    TEST_TEMPERATURES_F,
    TRAINING_DATA,
    Neuron,
    train,
    true_celsius,
)

PAGE = Path(__file__).with_name("web_ui.html")

# Limits so a typo in the browser can't make the server work forever.
MAX_EPOCHS = 100_000
# Long runs are thinned out so the page stays fast: every early epoch is
# kept (that's where the interesting changes happen), then evenly spaced
# samples up to roughly this many points in total.
MAX_HISTORY_POINTS = 2_000
KEEP_ALL_EPOCHS_UP_TO = 500
# Number of cells per side in the loss landscape grid.
LANDSCAPE_SIZE = 70


def snapshot(epoch, neuron):
    """Record the neuron's state at one moment in training."""
    grad_w, grad_b = neuron.gradients(TRAINING_DATA)
    return {
        "epoch": epoch,
        "w": neuron.w,
        "b": neuron.b,
        "loss": neuron.loss(TRAINING_DATA),
        "grad_w": grad_w,
        "grad_b": grad_b,
    }


def run_training(learning_rate, epochs, normalize):
    """Train a fresh neuron and return its whole history.

    This is the same training as the command-line program; the only
    difference is that we save a snapshot after each epoch instead of
    drawing ASCII art.
    """
    neuron = Neuron(input_scale=INPUT_SCALE if normalize else 1.0)
    step = max(1, math.ceil(epochs / MAX_HISTORY_POINTS))

    history = [snapshot(0, neuron)]  # epoch 0: before any learning

    def record(epoch, n):
        if epoch <= KEEP_ALL_EPOCHS_UP_TO or epoch % step == 0 or epoch == epochs:
            history.append(snapshot(epoch, n))

    exploded_at = train(neuron, TRAINING_DATA, learning_rate, epochs, on_epoch=record)

    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "input_scale": neuron.input_scale,
        "training_data": TRAINING_DATA,
        "test_data": [(f, true_celsius(f)) for f in TEST_TEMPERATURES_F],
        "history": history,
        "exploded_at": exploded_at,
        "landscape": loss_landscape(neuron.input_scale),
    }


def loss_landscape(input_scale):
    """Compute the loss for a grid of (w, b) values.

    This is the "valley" that gradient descent walks down. Each cell holds
    the loss the neuron would have with that particular w and b. The page
    colors the grid and draws training's path across it.

    The grid is centered between the starting point (0, 0) and the best
    possible answer, so the whole journey fits on the picture.
    """
    best_w = 5 / 9 * input_scale     # the slope, as the neuron sees it
    best_b = -32 * 5 / 9             # the offset
    w_pad = abs(best_w) * 0.6
    b_pad = abs(best_b) * 1.6
    w_min, w_max = min(0, best_w) - w_pad, max(0, best_w) + w_pad
    b_min, b_max = min(0, best_b) - b_pad, max(0, best_b) + b_pad

    probe = Neuron(input_scale=input_scale)
    rows = []
    for j in range(LANDSCAPE_SIZE):               # rows: b from top (max) to bottom (min)
        probe.b = b_max - (b_max - b_min) * j / (LANDSCAPE_SIZE - 1)
        row = []
        for i in range(LANDSCAPE_SIZE):           # columns: w from left (min) to right (max)
            probe.w = w_min + (w_max - w_min) * i / (LANDSCAPE_SIZE - 1)
            row.append(probe.loss(TRAINING_DATA))
        rows.append(row)

    return {
        "w_min": w_min, "w_max": w_max,
        "b_min": b_min, "b_max": b_max,
        "best_w": best_w, "best_b": best_b,
        "loss": rows,
    }


class Handler(BaseHTTPRequestHandler):
    """Answers two kinds of requests: the page itself, and /api/train."""

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self.send(200, "text/html; charset=utf-8", PAGE.read_bytes())
        elif url.path == "/api/train":
            self.handle_train(parse_qs(url.query))
        else:
            self.send(404, "text/plain", b"Not found")

    def handle_train(self, query):
        try:
            learning_rate = float(query.get("lr", ["0.1"])[0])
            epochs = int(query.get("epochs", ["200"])[0])
            normalize = query.get("normalize", ["1"])[0] == "1"
            if not (learning_rate > 0 and math.isfinite(learning_rate)):
                raise ValueError("learning rate must be a positive number")
            if not 1 <= epochs <= MAX_EPOCHS:
                raise ValueError(f"epochs must be between 1 and {MAX_EPOCHS:,}")
        except ValueError as e:
            body = json.dumps({"error": str(e)}).encode()
            self.send(400, "application/json", body)
            return

        result = run_training(learning_rate, epochs, normalize)
        self.send(200, "application/json", json.dumps(result).encode())

    def send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # keep the terminal quiet


def main():
    parser = argparse.ArgumentParser(
        description="Watch a single neuron learn F -> C in your web browser.")
    parser.add_argument("--port", type=int, default=8000,
                        help="port for the local web server (default 8000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="don't open a browser window automatically")
    args = parser.parse_args()

    # 127.0.0.1 means only this computer can reach the server.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"FtoCNeuralNet web UI running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
