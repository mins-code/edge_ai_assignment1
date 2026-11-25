"""Black-box tests over real HTTP plus independent artifact and accounting checks."""

import contextlib
import json
import math
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

from app.artifacts import MODEL_DIR, load_model
from app.model import BatteryNet, describe

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]
TYPICAL = dict(zip(FIELDS, [3.86, 28.0, 5, -66, 22]))


@contextlib.contextmanager
def server():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-access-log",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 45
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    if requests.get(url + "/health", timeout=1).ok:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(0.08)
            else:
                log.seek(0)
                raise RuntimeError("Uvicorn failed to start: " + log.read())
            yield url
        finally:
            proc.terminate()
            proc.wait(timeout=10)


def synthetic(n):
    rng = np.random.default_rng(2026)
    columns = [
        rng.uniform(3.0, 4.3, n),
        rng.uniform(-12, 55, n),
        rng.integers(0, 65, n),
        rng.integers(-99, -29, n),
        rng.uniform(0, 55, n),
    ]
    return [dict(zip(FIELDS, row)) for row in np.array(columns).T.tolist()]


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = cls.enterClassContext(server())
        cls.http = cls.enterClassContext(requests.Session())
        cls.config = json.loads((MODEL_DIR / "config.json").read_text())

    def post(self, path, payload):
        return self.http.post(self.url + path, json=payload, timeout=10)

    def ok(self, path, payload):
        response = self.post(path, payload)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body), {"days_left", "timings_ms"})
        stages = body["timings_ms"]
        self.assertEqual(set(stages), {"parse", "normalise", "inference", "build", "server_total"})
        for value in stages.values():
            self.assertIs(type(value), float)
            self.assertTrue(math.isfinite(value) and value >= 0)
        self.assertAlmostEqual(
            sum(stages[k] for k in stages if k != "server_total"), stages["server_total"], places=7
        )
        return body["days_left"]

    def rejected(self, response):
        self.assertEqual(response.status_code, 422, response.text[:300])
        body = response.json()
        self.assertNotIn("days_left", body)
        self.assertTrue(body["detail"])
        self.assertTrue(body["detail"][0]["msg"])

    def test_health(self):
        health = self.http.get(self.url + "/health").json()
        self.assertEqual(set(health), {"status", "team", "params", "val_mae"})
        self.assertEqual((health["status"], health["team"]), ("ok", 2))
        self.assertGreater(health["val_mae"], 0)
        self.assertEqual(health["val_mae"], self.config["val_mae"])

    def test_model_info_from_weight_shapes(self):
        state = torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True)
        layers = []
        prev = 5
        for key, weight in state.items():
            if key.endswith(".weight"):
                b, a = weight.shape
                self.assertEqual(a, prev)
                prev = b
                bias = state[key.replace(".weight", ".bias")]
                self.assertEqual(bias.shape, (b,))
                self.assertEqual(weight.dtype, torch.float32)
                count = weight.numel() + bias.numel()
                layers.append(
                    {
                        "name": f"linear{len(layers) + 1}",
                        "in": a,
                        "out": b,
                        "macs": a * b,
                        "params": count,
                        "bytes_f32": count * 4,
                    }
                )
        self.assertEqual(prev, 1)
        count = sum(x.numel() for x in state.values())
        expected = {
            "layers": layers,
            "total_params": count,
            "total_macs": sum(v["macs"] for v in layers),
            "total_bytes_f32": count * 4,
            "total_bytes_int8": count,
            "fits_300kb_sram_f32": count * 4 <= 307200,
            "fits_300kb_sram_int8": count <= 307200,
        }
        self.assertEqual(self.http.get(self.url + "/model_info").json(), expected)
        self.assertEqual(self.http.get(self.url + "/health").json()["params"], count)

    def test_held_out_single_http_accuracy(self):
        raw = pd.read_csv(ROOT / "data/team_02_train.csv")
        # Reconstruct the cleaned dataset independently from the three observed sentinel codes.
        good = raw[(raw.battery_v != 0) & (raw.temp_c != -127) & (raw.wifi_rssi_dbm != 0)]
        self.assertEqual(len(good), 1976)
        ids = np.random.default_rng(0).permutation(len(good))
        val, train = good.iloc[ids[:395]], good.iloc[ids[395:]]
        x = train[FIELDS].to_numpy(np.float32)
        np.testing.assert_allclose(self.config["mu"], x.mean(0), rtol=1e-6)
        np.testing.assert_allclose(self.config["sd"], x.std(0), rtol=1e-6)
        self.assertEqual(self.config["features"], FIELDS)
        pred = [self.ok("/predict", row) for row in val[FIELDS].to_dict("records")]
        mae = np.abs(np.array(pred) - val.days_left).mean()
        self.assertAlmostEqual(mae, self.config["val_mae"], delta=0.0051)
        self.assertLess(mae, 6.0)
        print(f"HTTP validation MAE {mae:.6f} days on {len(pred)} labeled readings", flush=True)

    def test_boundaries_and_out_of_distribution(self):
        for row in (
            TYPICAL,
            dict(zip(FIELDS, [2.0, -40.0, 0.0, -120.0, 0.0])),
            dict(zip(FIELDS, [4.5, 85.0, 86400.0, -0.01, 240.0])),
            dict(zip(FIELDS, [4.24, 49, 46.5, -91, 38.5])),
        ):
            with self.subTest(row=row):
                days = self.ok("/predict", row)
                self.assertTrue(
                    isinstance(days, (int, float)) and math.isfinite(days) and days >= 0
                )

    def test_bad_readings(self):
        for field, values in zip(
            FIELDS, [[0, -1, 1.9, 4.6], [-127, -41, 86], [-1, 86401], [-121, 0, 1], [-1, 241]]
        ):
            for value in [*values, None, True, [], {}, "3.8"]:
                with self.subTest(field=field, value=value):
                    self.rejected(self.post("/predict", {**TYPICAL, field: value}))

    def test_missing_extra_wrong_shapes(self):
        for field in FIELDS:
            self.rejected(self.post("/predict", {k: v for k, v in TYPICAL.items() if k != field}))
        for body in ({**TYPICAL, "extra": 1}, [], "text", 0, {}):
            self.rejected(self.post("/predict", body))

    def test_malformed_and_nonfinite(self):
        for body in (
            '{"',
            "",
            "null",
            *[
                json.dumps({**TYPICAL, "temp_c": v})
                for v in (float("nan"), float("inf"), -float("inf"))
            ],
            json.dumps(TYPICAL).replace("28.0", "1e999"),
        ):
            self.rejected(
                self.http.post(
                    self.url + "/predict", data=body, headers={"Content-Type": "application/json"}
                )
            )

    def test_invalid_batches(self):
        for body in (
            {},
            {"rows": []},
            {"rows": None},
            {"rows": TYPICAL},
            {"rows": [TYPICAL, None]},
            {"rows": [TYPICAL, {**TYPICAL, "age_months": -1}]},
            {"rows": [TYPICAL], "extra": 1},
            {"rows": [TYPICAL] * 10001},
        ):
            self.rejected(self.post("/predict_batch", body))

    def test_order_and_batch_sizes(self):
        rows = synthetic(400)
        singles = [self.ok("/predict", row) for row in rows]
        for n in (1, 50, 400):
            batch = self.ok("/predict_batch", {"rows": rows[:n]})
            self.assertEqual(batch, singles[:n])
            self.assertTrue(all(math.isfinite(v) and v >= 0 for v in batch))

    def test_largest_supported_batch(self):
        row = self.ok("/predict", TYPICAL)
        self.assertEqual(self.ok("/predict_batch", {"rows": [TYPICAL] * 10000}), [row] * 10000)

    def test_independent_numpy_forward(self):
        weights = torch.load(MODEL_DIR / "model.pt", weights_only=True)
        rows = synthetic(50)
        x = np.array([[row[f] for f in FIELDS] for row in rows], dtype=float)
        x = (x - self.config["mu"]) / self.config["sd"]
        matrices = [k for k in weights if k.endswith(".weight")]
        for i, key in enumerate(matrices):
            x = x @ weights[key].numpy().T + weights[key.replace(".weight", ".bias")].numpy()
            if i < len(matrices) - 1:
                x = np.maximum(x, 0)
        served = self.ok("/predict_batch", {"rows": rows})
        np.testing.assert_allclose(served, x[:, 0].clip(min=0), atol=0.0052, rtol=0)

    def test_repeat_and_restart(self):
        first = self.ok("/predict", TYPICAL)
        self.assertEqual(self.ok("/predict", TYPICAL), first)
        with server() as second:
            self.assertEqual(
                requests.post(second + "/predict", json=TYPICAL).json()["days_left"], first
            )

    def test_interactive_docs(self):
        spec = self.http.get(self.url + "/openapi.json").json()
        for path in ("/predict", "/predict_batch"):
            self.assertIn("application/json", spec["paths"][path]["post"]["requestBody"]["content"])


class ArtifactTests(unittest.TestCase):
    def test_mismatched_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            shutil.copy(MODEL_DIR / "model.pt", dest)
            cfg = json.loads((MODEL_DIR / "config.json").read_text())
            for key, value in (
                ("hidden", [8]),
                ("mu", [0] * 5),
                ("features", FIELDS[::-1]),
                ("sd", [0] * 5),
                ("val_mae", 0.0),
            ):
                (dest / "config.json").write_text(json.dumps({**cfg, key: value}))
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    load_model(dest)
            shutil.copy(MODEL_DIR / "config.json", dest)
            with (dest / "model.pt").open("ab") as f:
                f.write(b"x")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_model(dest)

    def test_resource_thresholds(self):
        for width in (32, 128, 256, 512, 2048, 273, 274, 550, 551):
            info = describe(BatteryNet((width, width)))
            params = width**2 + 8 * width + 1
            self.assertEqual(info["total_params"], params)
            self.assertEqual(info["total_macs"], width**2 + 6 * width)
            self.assertEqual(info["fits_300kb_sram_int8"], params <= 307200)
            self.assertEqual(info["fits_300kb_sram_f32"], 4 * params <= 307200)


if __name__ == "__main__":
    unittest.main()
