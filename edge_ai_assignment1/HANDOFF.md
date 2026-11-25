# Team 02 · Battery life API handoff

Start with the local test suite after a code change. Before the grading window, run the public smoke test from a laptop outside the Codespace. The suite starts its own Uvicorn process, while the smoke test reaches the deployed server through GitHub's public port.

## Last verified state

The November 20, 2026 public check at 11:04 IST covered 419 anonymous requests. All 405 valid requests returned HTTP 200 and all 14 invalid requests returned HTTP 422. Batches of 1, 50 and 400 preserved order and matched individual predictions exactly. All 400 synthetic predictions matched the checkpoint from commit `d2ff4c5` to two decimals. Warm public round-trip latency was 48 ms at p50 and 133 ms at p95. The Codespace's idle timeout and future availability still need an owner check. [Public health endpoint](https://didactic-barnacle-x59grjp447w4cv676-8000.app.github.dev/health)

The local baseline is **15 passing tests and 3.9784 days MAE** over 395 held-out Team 02 readings sent through the API. The synthetic public checks verified deployment consistency. Accuracy on the professor's hidden readings is unknown. [Test output](reports/tests.txt) · [Evaluation checklist](reports/CHECKLIST.md) · [Training and benchmark report](reports/REPORT.md)

## Run the local suite

From the repository root, use Python 3.11. In a new local checkout, run the following commands.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r edge_ai_assignment1/requirements.txt
cd edge_ai_assignment1
python -m unittest discover -s tests -v
```

In the supplied Codespace, the root devcontainer installs dependencies. Run `cd edge_ai_assignment1` and then run `python -m unittest discover -s tests -v` directly. To narrow a failure, add a name filter, for example `python -m unittest discover -s tests -v -k batch`. The suite starts and stops its own servers and loads the committed model. To explore the API locally, run `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000` from `edge_ai_assignment1/` and open [local Swagger docs](http://127.0.0.1:8000/docs).

## Smoke-test the public deployment

Run this from an external terminal with the project's Python environment active. Substitute the forwarded address if the Codespace has been recreated.

```sh
export BATTERY_API_URL='https://didactic-barnacle-x59grjp447w4cv676-8000.app.github.dev'
python - <<'PY'
import os
import requests

url = os.environ["BATTERY_API_URL"].rstrip("/")
row = dict(battery_v=3.86, temp_c=28, rings_per_day=5, wifi_rssi_dbm=-66, age_months=22)
options = dict(timeout=30, allow_redirects=False)
with requests.Session() as s:
    s.trust_env = False
    health = s.get(url + "/health", **options)
    assert health.status_code == 200, (health.status_code, health.text[:200])
    h = health.json()
    assert (h["status"], h["team"], h["params"]) == ("ok", 2, 1281), h
    info = s.get(url + "/model_info", **options)
    assert info.status_code == 200, info.status_code
    m = info.json()
    assert (m["total_params"], m["total_bytes_f32"], m["total_bytes_int8"], m["total_macs"]) == (1281, 5124, 1281, 1216), m
    one = s.post(url + "/predict", json=row, **options)
    assert one.status_code == 200, one.text
    prediction = one.json()
    assert prediction["days_left"] == 115.48, prediction
    batch = s.post(url + "/predict_batch", json={"rows": [row]}, **options)
    assert batch.status_code == 200 and batch.json()["days_left"] == [115.48], batch.text
    bad = s.post(url + "/predict", json={**row, "battery_v": 0}, **options)
    assert bad.status_code == 422, bad.text
    print("PASS", h, prediction)
PY
```

For a public latency sample, run `python client/bench.py --url "$BATTERY_API_URL" --n 200`. Add `--no-keepalive` for a fresh connection per prediction. Each run writes a timestamped CSV into the current directory.

## Restore service before grading

1. Open the deployed Codespace, `cd edge_ai_assignment1`, and check `curl -fsS http://localhost:8000/health`. If the server is down, start it from `edge_ai_assignment1/` with `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`. Leave that process running.
2. In **Ports**, set port **8000 → Port Visibility → Public** and copy its forwarded HTTPS address. If local health succeeds but the public test redirects to sign-in, recheck visibility and the copied URL.
3. Run the external smoke test again. Give the instructor the verified **base URL** and the repository link. The Codespace owner should check the idle timeout and confirm reachability again when the grading window opens. [Team repository](https://github.com/mins-code/edge_ai_assignment1)

## Continue development

| If you are changing… | Start here |
|---|---|
| Request validation, timing or batch behavior | [Routes](app/main.py), [shared sensor bounds](app/sensors.py), [HTTP regression tests](tests/test_service.py) |
| Network layers, parameter counts or checkpoint loading | [Network and accounting](app/model.py), [artifact loader](app/artifacts.py), [saved config](model/config.json) |
| Data cleaning, model selection or training | [Cleaning and splits](training_data.py), [fold experiments](reports/experiment.py), [frozen recipe](reports/recipe.json), [trainer](train.py) |

For model work, use only `data/team_02_train.csv`. Fit preprocessing on training rows and keep the 395-row holdout out of model selection. Reproduce the selected fit with `python train.py --team 2` on a development branch. This overwrites both `model/model.pt` and `model/config.json`; the loader checks their fingerprint, so publish the matching pair together. After an intentional model change, review the expected prediction and resource counts in this handoff, rerun the suite, restart Uvicorn on the new revision, and repeat the external check.
