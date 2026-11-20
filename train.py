"""Fit the Team 02 battery regressor and export a CPU inference checkpoint."""

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from app.artifacts import MODEL_DIR, fingerprint
from app.model import FEATURES, BatteryNet
from training_data import DATA, baselines, clean, split


@dataclass(frozen=True)
class Options:
    hidden: tuple = (32, 32)
    lr: float = 0.006
    epochs: int = 1200
    seed: int = 41
    loss: str = "mse"
    decay: float = 0.0
    normalize: bool = True


def arrays(frame):
    return frame[FEATURES].to_numpy(np.float32), frame.days_left.to_numpy(np.float32)


def fit(train, opt, monitor=None):
    """A monitor is supplied only for inner folds. The final fit runs a fixed epoch count."""
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(opt.seed)
    net = BatteryNet(opt.hidden)
    x, y = arrays(train)
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1
    if not opt.normalize:
        mu.fill(0)
        sd.fill(1)
    xt = torch.from_numpy((x - mu) / sd)
    center, scale = float(y.mean()), float(y.std()) or 1.0
    yt = torch.from_numpy((y - center) / scale)
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.decay)
    if monitor is not None:
        xv, yv = arrays(monitor)
        xv = torch.from_numpy((xv - mu) / sd)
    best, best_epoch = float("inf"), opt.epochs
    for epoch in range(1, opt.epochs + 1):
        pred = net(xt)
        if opt.loss == "huber":
            loss = F.smooth_l1_loss(pred, yt, beta=5 / scale)
        else:
            loss = F.mse_loss(pred, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if monitor is not None and (epoch % 20 == 0 or epoch == opt.epochs):
            with torch.inference_mode():
                score = np.abs((net(xv) * scale + center).clamp_min(0).numpy() - yv).mean()
            if score < best:
                best, best_epoch = score, epoch
                state = copy.deepcopy(net.state_dict())
            if epoch - best_epoch >= 240:
                break
    if monitor is not None:
        net.load_state_dict(state)
    # Absorb the affine target transform into the output layer. The saved net returns days.
    with torch.no_grad():
        net.net[-1].weight.mul_(scale)
        net.net[-1].bias.mul_(scale).add_(center)
    stats = {
        **asdict(opt),
        "params": sum(p.numel() for p in net.parameters()),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "seconds": time.perf_counter() - started,
    }
    return net.eval(), mu, sd, stats


@torch.inference_mode()
def score(net, mu, sd, frame):
    x, y = arrays(frame)
    pred = net(torch.from_numpy((x - mu) / sd)).clamp_min(0).numpy()
    return float(np.abs(pred.astype(float) - y).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team", type=int, required=True, choices=[2])
    # The sweep writes a frozen recipe before the final holdout is evaluated.
    recipe = json.loads((Path(__file__).parent / "reports/recipe.json").read_text())
    ap.add_argument("--hidden", default=",".join(map(str, recipe["hidden"])))
    ap.add_argument("--epochs", type=int, default=recipe["epochs"])
    args = ap.parse_args()
    recipe.update(hidden=tuple(map(int, args.hidden.split(","))), epochs=args.epochs)
    if args.epochs < 1 or any(w < 1 for w in recipe["hidden"]):
        ap.error("epochs and hidden widths must be positive")
    raw = pd.read_csv(DATA)
    good, _ = clean(raw)
    train, val = split(good)
    net, mu, sd, metrics = fit(train, Options(**recipe))
    metrics["train_mae"] = score(net, mu, sd, train)
    metrics["val_mae"] = score(net, mu, sd, val)
    metrics["gap"] = metrics["val_mae"] - metrics["train_mae"]
    mean, linear = baselines(train, val)
    print(f"{len(raw)} rows, {len(raw) - len(good)} glitch rows dropped")
    print("Mean baseline:", np.abs(mean - val.days_left).mean())
    print("Linear baseline:", np.abs(linear - val.days_left).mean())
    config = dict(
        team=args.team,
        features=FEATURES,
        mu=mu.tolist(),
        sd=sd.tolist(),
        split_seed=0,
        train_rows=len(train),
        val_rows=len(val),
        **metrics,
    )
    MODEL_DIR.mkdir(exist_ok=True)
    weights = MODEL_DIR / "model.pt"
    torch.save(net.state_dict(), weights)
    config["fingerprint"] = fingerprint(config, weights)
    (MODEL_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
