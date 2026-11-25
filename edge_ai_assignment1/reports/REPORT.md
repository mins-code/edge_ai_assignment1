# How long will the battery last? · Team 02

**The shipped API missed by 3.9784 days on average across 395 held-out readings sent individually over HTTP.** Linear regression missed by 19.7314 days on the same rows. The 5 → 32 → 32 → 1 network uses 5,124 bytes of float32 parameters. Both the weights and the input statistics ship in Git, so the server starts directly from the checkpoint.

## Data and cleaning

The repository supplies one team CSV with 2,000 rows. It has zero missing cells and zero duplicate rows. Removing eight zero-voltage records, six −127 °C records and ten zero-RSSI records leaves **1,976 labeled readings**. Each removed row had one glitch. Zero volts cannot power this reporting device, −127 °C is an isolated sensor fault code in this outdoor application, and zero RSSI is treated as a missing associated-link reading. [Full column statistics](data-summary.txt) · [Raw histograms](histograms.png)

| Column | CSV type | Clean range | Median | Accepted service range |
|---|---|---:|---:|---|
| Battery voltage (V) | float | 3.35–4.20 | 3.763 | 2.0–4.5 |
| Temperature (°C) | float | 8–46 | 28.3 | −40–85 |
| Rings per day | integer | 0–45 | 7 | 0–86,400 |
| Wi-Fi RSSI (dBm) | integer | −88 to −45 | −66 | −120 up to, excluding, 0 |
| Device age (months) | integer | 0–36 | 18 | 0–240 |
| Remaining life (days) | float | 0.5–271.8 | 49.1 | Training labels must be finite and nonnegative |

The service limits assume a powered single-cell device, outdoor electronics, at most one recorded ring per second and a 20-year service life. These are engineering assumptions. Numeric fractional rates and ages are accepted. Zero rings, a new device, the warmest observations, weak signals and long-lived batteries were all retained. Finite values outside the observed range can pass validation, although accuracy there has less supporting evidence.

The target is right-skewed, with a mean of 63.3 days and an interquartile range of 16.6–98.8 days. On the development rows, voltage rises with remaining life (Pearson r = 0.75). The response steepens near the middle of the charge range, then flattens. Stronger Wi-Fi correlates with longer life (0.39), while frequent rings correlate with shorter life (−0.24). Age (−0.12) and temperature (−0.06) have weaker marginal associations. Absolute correlations between input pairs are at most 0.05 after rounding. At high voltage, readings with similar charge can have very different lifetimes. In the development baseline split, the linear fit underpredicted the longest-lived quartile by 19.54 days on average, with MAE of 24.54 days in that quartile. [Feature relationships and linear predictions](relationships.png)

## Selection and accuracy

A NumPy permutation with seed 0 assigns 395 clean rows to the final holdout and 1,581 to development. The final holdout supplied no gradients, preprocessing statistics, stopping decisions or architecture choices. A separate seeded split within development yielded initial mean and least-squares baseline MAEs of 44.74 and 18.25 days. For a reference run, the original trainer ran unmodified on the raw CSV with the final holdout rows withheld. Its own split included glitches and produced 23.29 days validation MAE after 1,500 epochs. Its split differs from the table below. [Starter run](starter.txt)

For selection, all candidates used the same three disjoint validation folds inside development (fold partition seed 31). The width sweep used Adam, learning rate 0.006, MSE, up to 1,200 full-batch epochs and initialization seeds 41–43 across folds. Every 20 epochs, the trainer checked validation MAE and saved the best weights. It stopped after 240 epochs without improvement. Each training fold fitted its own input and target statistics.

| Hidden width | One layer params | One layer mean fold MAE | Two layers params | Two layers mean fold MAE |
|---:|---:|---:|---:|---:|
| 4 | 29 | 9.801 | 49 | 6.675 |
| 8 | 57 | 5.967 | 129 | 4.552 |
| 16 | 113 | 5.355 | 385 | 3.988 |
| 32 | 225 | 4.661 | 1,281 | 3.877 |
| 64 | 449 | 4.402 | 4,609 | 3.951 |

The two best base architectures received three follow-up recipes. These covered learning rates 0.003 and 0.009, epoch ceilings of 2,000 and 3,200, seeds 71–73, Huber-style Smooth L1 loss and weight decay of 0.00001. The 32 × 32 MSE control without input normalization scored 17.416 days, compared with 3.877 under the same normalized training protocol. Small widths underfit. Increasing from 32 to 64 units in both layers more than tripled the parameter count without improving mean validation error. [MAE versus parameters](architecture.png) · [All 51 fold runs, including training MAE, validation MAE, gap, duration and epoch](experiments.csv)

Selection favored the smallest candidate within 0.05 day of the best mean, then the lowest MAE at that size. The winning 32 × 32 recipe used Smooth L1 with a five-day transition, learning rate 0.003 and weight decay 0.00001. Its selection-fold MAEs ranged from 3.764 to 4.025, averaging 3.871. Its advantage over the MSE recipe was only 0.006 day. The median best epoch across folds was 320. The final run used seed 41 and exactly 320 epochs on all development rows. [Frozen training recipe](recipe.json)

| Final evaluation | MAE in days |
|---|---:|
| Training-set mean on final holdout | 45.5161 |
| Least-squares linear model on final holdout | 19.7314 |
| Final network on training rows | 2.8810 |
| Final network on holdout, full precision | 3.9783 |
| Actual single-request API on holdout, rounded to two decimals | 3.9784 |

The final fit took 0.885 seconds on the training machine. The 1.097-day train-to-holdout gap shows measurable overfitting. Across comparable unseen readings, **about four days MAE** is a reasonable expectation. Resampling 400 absolute errors with replacement from the 395 holdout errors gives a central 95% range of 3.53–4.45 days across 3,000 draws (seed 20). That calculation assumes the hidden readings resemble the holdout. The professor's readings and labels are unavailable. [Saved configuration and metrics](../model/config.json)

## How one reading becomes a prediction

1. FastAPI validates the five named JSON numbers, rejects unexpected fields and returns 422 with the field and reason for an invalid reading. Batch validation applies the same schema to each row before inference.
2. For each feature, subtract its training mean and divide by its training standard deviation. This puts volts, months and dBm on comparable numerical scales. The saved feature order determines the matrix columns.
3. Each hidden layer computes weighted sums plus biases and applies ReLU, which sets negative hidden activations to zero. The final linear layer combines the 32 hidden values into one estimate in days. Training used standardized targets to keep gradients manageable. Export folds that scale and offset into the last layer's weights and bias.
4. During training, backpropagation computes loss gradients and Adam updates the weights. Smooth L1 is quadratic for small errors and grows linearly for larger ones. During serving, the model runs in inference mode, clamps negative day estimates to zero and rounds to two decimals.
5. Single and batch routes use the same float32 forward pass in 32-row blocks. Padding the last block keeps arithmetic consistent at rounding boundaries. Padded outputs are discarded. The config fingerprint detects mismatched weights or metadata at startup.

## Model memory and computation

A Linear layer from a inputs to b outputs stores ab weights and b biases and performs ab multiply-accumulates (MACs) per reading. ReLU has no trainable parameters. The hand calculation gives the following table, independently checked against the serialized tensor shapes.

| Layer | Weights + biases | Parameters | Float32 bytes | MACs |
|---|---|---:|---:|---:|
| 5 → 32 | 160 + 32 | 192 | 768 | 160 |
| 32 → 32 | 1,024 + 32 | 1,056 | 4,224 | 1,024 |
| 32 → 1 | 32 + 1 | 33 | 132 | 32 |
| **Total** | **1,216 + 65** | **1,281** | **5,124** | **1,216** |

The assignment's int8 convention gives 1,281 parameter bytes. For a one-reading embedded implementation, the largest individual activation is 32 elements (128 float32 bytes). Keeping both layer input and output live at the middle layer requires 64 elements (256 float32 bytes, or 64 int8 bytes with in-place ReLU). The ten float32 normalization constants take another 40 bytes. The idealized float32 total for parameters, two activation buffers and normalization is **5,420 bytes**. Frameworks, stacks, alignment, quantization scales and wider accumulators require additional memory. The checkpoint contains float32 weights. Int8 sizes describe hypothetical storage.

For 5 → w → w → 1, parameters = w² + 8w + 1 and MACs = w² + 6w. The table uses the service's network inspection function on each width. Fit decisions here apply to parameters alone, using 307,200 bytes free SRAM and 8,388,608 bytes PSRAM.

| w | Parameters / int8 bytes | Float32 bytes | SRAM f32 / int8 | PSRAM f32 / int8 | MACs |
|---:|---:|---:|---|---|---:|
| 32 | 1,281 | 5,124 | yes / yes | yes / yes | 1,216 |
| 128 | 17,409 | 69,636 | yes / yes | yes / yes | 17,152 |
| 256 | 67,585 | 270,340 | yes / yes | yes / yes | 67,072 |
| 512 | 266,241 | 1,064,964 | no / yes | yes / yes | 265,216 |
| 2,048 | 4,210,689 | 16,842,756 | no / no | no / yes | 4,206,592 |

Float32 first exceeds free SRAM at **w = 274** (309,076 bytes, up from 306,856 at 273). Int8 first exceeds it at **w = 551** (308,010 bytes, up from 306,901 at 550). The 2,048-wide float32 weights alone also exceed the assignment's 16 MiB flash budget.

**ESP32-S3 estimate.** At 240 MHz on one core, assuming 10 cycles per MAC for a simple compiled kernel, 1,216 MACs would take about **51 microseconds**. A 5–20 cycle assumption gives 25–101 microseconds for the MACs. The clock comes from Espressif. Cycles per MAC are an illustrative engineering assumption. A complete prediction also pays for normalization, bias, activation, memory access and runtime overhead. Board latency, quantized accuracy and energy consumption require hardware measurements. [Espressif ESP32-S3 specifications](https://www.espressif.com/en/products/socs/esp32-s3)

For the doorbell, local inference would keep usage and radio telemetry on-device, avoid a radio wake and server charge for each prediction, and work during Wi-Fi outages. The model's small memory requirement makes that practical. The server simplifies centralized model updates and meets the assignment's grading interface. Local benchmarks below already show request overhead dominates the model calculation.

## Measured API latency

The supplied client ran unchanged twice against Uvicorn on macOS arm64 loopback, 200 predictions per run. The first prediction took **9.109 ms** with keep-alive and **1.714 ms** in the later connection-close run. Each run started with a health request. Only the first run made the server's first prediction after startup. Warm percentiles exclude the first prediction in each run.

| Milliseconds | Keep p50 | Keep p95 | Keep p99 | Close p50 | Close p95 | Close p99 |
|---|---:|---:|---:|---:|---:|---:|
| Round trip | 0.845 | 0.979 | 1.132 | 1.060 | 1.438 | 1.653 |
| Parse and validation | 0.042 | 0.053 | 0.078 | 0.046 | 0.069 | 0.092 |
| Normalise | 0.006 | 0.009 | 0.015 | 0.008 | 0.019 | 0.032 |
| Inference | 0.045 | 0.052 | 0.084 | 0.051 | 0.089 | 0.113 |
| Build result | 0.001 | 0.002 | 0.002 | 0.001 | 0.002 | 0.003 |
| Server total | 0.095 | 0.112 | 0.173 | 0.106 | 0.179 | 0.234 |
| Network, framework and client remainder | 0.750 | 0.872 | 1.000 | 0.952 | 1.270 | 1.432 |

The pre-benchmark estimate was 0.1 ms for inference and 100 ms for the laptop-to-Codespaces round trip. Framework and transport overhead were expected to dominate. Warm local inference measured 0.045 ms at the median. Even over loopback, about 0.75 ms elapsed outside the timed handler, including client work, routing, serialization and transport. Closing the connection added about 0.22 ms to the median round trip. The public Codespaces path still needs a separate measurement. [Time breakdown for one representative request](bench/stages.png) · [Keep-alive samples](bench/keepalive.csv) · [Connection-close samples](bench/close.csv)

For the same 400 rows, six paired trials alternated execution order and checked every prediction for equality. Median total time was **346.57 ms** for 400 sequential requests over a persistent session and **2.80 ms** for one batch, about **124× faster**. One batch amortizes HTTP, validation dispatch and tensor setup over many readings. [Paired trials](bench/batch.csv)

## Running locally and in Codespaces

Run from the repository root with Python 3.11. Use these commands for a new local environment.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

After publishing the verified Git revision, open **Code → Codespaces** on the repository and create a Codespace on that revision. The supplied devcontainer installs dependencies. Start the server from the repository root.

```sh
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a second Codespace terminal, run the internal benchmark. If initial dependency setup was interrupted, run `python -m pip install -r requirements.txt` first.

```sh
curl -fsS http://localhost:8000/health
python client/bench.py --url http://localhost:8000 --n 200
```

In **Ports → 8000 → Port Visibility**, select **Public**. Copy the forwarded HTTPS address. On the laptop, with the local environment active, substitute the copied address in the commands below.

```sh
URL='https://YOUR-CODESPACE-NAME-8000.app.github.dev'
curl -fsS "$URL/health"
curl -fsS "$URL/model_info"
curl -fsS -H 'Content-Type: application/json' "$URL/predict" \
  -d '{"battery_v":3.86,"temp_c":28,"rings_per_day":5,"wifi_rssi_dbm":-66,"age_months":22}'
python client/bench.py --url "$URL" --n 200
python client/bench.py --url "$URL" --n 200 --no-keepalive
```

Check that the public health reply shows team 2 and 1,281 parameters. Open the address with `/docs` in a signed-out browser to verify public access. Submit the public base URL and repository link. Keep the Codespace active during the instructor's check window, start Uvicorn again after restart and recheck port visibility.

To reproduce data analysis, selection and training, run the following commands in a working copy. Training replaces the local model artifacts. Starting the API only reads them.

```sh
python -m reports.analyze
python -m reports.experiment
python train.py --team 2
python -m unittest discover -s tests -v
# With the API running in a separate terminal
python -m reports.benchmark --url http://127.0.0.1:8000
```

## Verification and remaining checks

**15 tests passed, 0 failed, 0 skipped** in a fresh local clone of implementation commit `69ae175`. An empty Python 3.11.15 environment on macOS arm64 installed `requirements.txt` with pip. Dependency checks passed, and Uvicorn started from the cloned repository using the committed artifacts. A subsequent independent 320-epoch refit took 1.176 seconds and reproduced every weight, bias, input mean and standard deviation exactly. It reproduced both reported MAEs. [Full test output](tests.txt) · [Instructor checks mapped to code and tests](CHECKLIST.md)

The suite scores 395 labeled single requests, independently computes the layer table from weight shapes, and evaluates the forward pass in NumPy. It covers the physical boundaries, plausible readings outside the training range, wrong types, missing and extra fields, NaN, infinities, malformed JSON, empty and mixed-validity batches, all five timing fields, and checkpoint/config mismatches. Predictions matched exactly between singles and ordered 50- and 400-row batches, and the 10,000-row limit passed. Restarting the server reproduced the same prediction. Three separate requests through the interactive Swagger UI for 0 V, −127 °C and 0 dBm each returned HTTP 422 and identified the field.

The benchmark evidence is local. Hosted Linux Codespaces startup, anonymous HTTPS reachability, public round-trip measurements and availability during the grading window require a published revision and running Codespace. No hidden labels or physical ESP32-S3 measurements were available.

## Changed files

| Purpose | Modified starter files | Created files |
|---|---|---|
| Training and serving | `train.py`, `app/main.py`, `app/model.py` | `training_data.py`, `app/sensors.py`, `app/artifacts.py` |
| Deployment | `requirements.txt`, `.gitignore` | `model/model.pt`, `model/config.json` |
| Automated verification | | `tests/test_service.py`, `reports/tests.txt`, `reports/CHECKLIST.md` |
| Data analysis | | `reports/analyze.py`, `reports/data-summary.txt`, `reports/histograms.png`, `reports/relationships.png` |
| Model selection | | `reports/experiment.py`, `reports/experiments.csv`, `reports/recipe.json`, `reports/architecture.png`, `reports/starter.txt` |
| Benchmark and report | | `reports/benchmark.py`, `reports/bench/keepalive.csv`, `reports/bench/close.csv`, `reports/bench/batch.csv`, `reports/bench/stages.png`, `reports/REPORT.md` |
