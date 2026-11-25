"""Check that model metadata and weights belong to the same training run."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from app.model import FEATURES, BatteryNet

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"


def fingerprint(config, weights):
    metadata = {k: v for k, v in config.items() if k != "fingerprint"}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    digest.update(weights.read_bytes())
    return digest.hexdigest()


def load_model(directory=MODEL_DIR):
    config = json.loads((directory / "config.json").read_text())
    weights = directory / "model.pt"
    if config.get("fingerprint") != fingerprint(config, weights):
        raise ValueError("Model and configuration fingerprint mismatch")
    mu, sd = (np.array(config[k], dtype=np.float32) for k in ("mu", "sd"))
    if (
        config["features"] != FEATURES
        or mu.shape != (5,)
        or sd.shape != (5,)
        or not np.isfinite(mu).all()
        or not np.isfinite(sd).all()
        or (sd <= 0).any()
        or not math.isfinite(config["val_mae"])
        or config["val_mae"] < 0
    ):
        raise ValueError("Invalid preprocessing or validation metadata")
    net = BatteryNet(config["hidden"])
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    if not all(torch.isfinite(p).all() for p in net.parameters()):
        raise ValueError("Non-finite checkpoint weights")
    return config, net.eval(), mu, sd
