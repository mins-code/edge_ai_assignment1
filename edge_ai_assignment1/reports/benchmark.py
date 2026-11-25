"""Run the original client and a paired 400-row benchmark against a running server."""

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from client.bench import reading

OUT = Path(__file__).resolve().parent / "bench"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for name in ("keepalive", "close"):
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable,
                str(OUT.parents[1] / "client/bench.py"),
                "--url",
                args.url,
                "--n",
                "200",
            ]
            if name == "close":
                command.append("--no-keepalive")
            result = subprocess.run(command, cwd=tmp, capture_output=True, text=True, check=True)
            (raw,) = Path(tmp).glob("bench_*.csv")
            frame = pd.read_csv(raw)
            if len(frame) != 200:
                raise RuntimeError("Benchmark did not return all 200 responses")
            (OUT / f"{name}.csv").write_text(raw.read_text())
            print(name, result.stdout)
    rng = np.random.default_rng(17)
    rows = [reading(rng) for _ in range(400)]
    timings = []
    with requests.Session() as session:

        def post(endpoint, data):
            response = session.post(args.url + endpoint, json=data, timeout=20)
            response.raise_for_status()
            return response.json()["days_left"]

        for trial in range(6):
            results = {}
            for mode in ("batch", "single") if trial % 2 else ("single", "batch"):
                start = time.perf_counter()
                if mode == "batch":
                    results[mode] = post("/predict_batch", {"rows": rows})
                else:
                    results[mode] = [post("/predict", row) for row in rows]
                timings.append(
                    {"trial": trial, "mode": mode, "ms": 1000 * (time.perf_counter() - start)}
                )
            if results["batch"] != results["single"]:
                raise AssertionError("400-row benchmark parity failed")
    pd.DataFrame(timings).to_csv(OUT / "batch.csv", index=False)
    for mode in ("single", "batch"):
        print(mode, "median ms", statistics.median(t["ms"] for t in timings if t["mode"] == mode))

    frame = pd.read_csv(OUT / "keepalive.csv").iloc[1:]
    example = frame.iloc[(frame.round_trip - frame.round_trip.median()).abs().argmin()]
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    keys = ["parse", "normalise", "inference", "build", "network_and_framework"]
    ax.barh([k.replace("_", " ") for k in keys], [example[k] for k in keys], color="#306684")
    ax.set(
        xlabel="Milliseconds",
        title=f"Local keep-alive request {int(example.i)} nearest median round trip",
    )
    fig.savefig(OUT / "stages.png", dpi=150)


if __name__ == "__main__":
    main()
