# Mission 3: Feature Extraction — Results & Analysis (Team 02)

## 1. Mathematical Pipeline & Optimization
- **Pipeline (`mission3_features.py` & `doorbell_board_starter/features.h`):**
  1. Frame `k`: samples `[k*hop, k*hop + win - 1]` (`win=480`, `hop=320` at `16 kHz` = `50 frames/s`).
  2. Symmetric Hann window (`0.5 - 0.5*cos(2*pi*n/(win-1))`), zero-padded to `nfft=512`.
  3. Power spectrum `P[k] = |X[k]|^2 / nfft` for `k = 0..256`.
  4. 40 HTK mel triangles (`20 Hz` to `8000 Hz`), `10*log10(max(m, 1e-10))`, optional orthonormal DCT-II (`mfcc=13`).
- **Optimization:** Replaced the $O(N^2)$ direct `dft()` (`123,360` `cosf`/`sinf` calls per frame) with an in-place **Radix-2 Cooley-Tukey FFT** using precomputed twiddle-factor (`tw_cos`, `tw_sin`) and bit-reversal (`bitrev`) tables in `begin()`, eliminating all trigonometric calls during `compute()`.
- **FeaturePipeline memory (`nfft=512, mels=40, mfcc=0`):** `window` (`1,920 B`), `re` (`2,048 B`), `im` (`2,048 B`), `pw` (`1,028 B`), `melw` (`41,120 B`), `melv` (`160 B`), `tw_cos` (`1,024 B`), `tw_sin` (`1,024 B`), `bitrev` (`2,048 B`) = **`52,416 B` (~51.2 KiB)** (plus `2,080 B` for `dct` when `mfcc=13`).

## 2. Baseline vs. Optimized Performance (`baseline_vs_optimized.png`)
*(Note: The baseline `dft()` 10-second `speed=1.0` benchmark on both `COM5` and `COM10` timed out after ~30.1 s with partial frames (`16 / 499` received, `483` lost) because its ~1.86 s/frame compute time is ~93× slower than the 20 ms hop budget.)*

| Version | Port / Runs | Frames (Recv/Exp) | Median `compute_us` | p95 `compute_us` | Max `compute_us` | Max Lag (s) | Max Abs Error |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline (`dft`, `baseline_perf.csv`) | `COM5` (10s, `timeout_partial`) | 16 / 499 (`483` lost) | 1,855,697.5 | 1,869,196.8 | 1,873,891.0 | 29.5074 | 0.00168 dB |
| Baseline (`dft`, `baseline_perf.csv`) | `COM10` (10s, `timeout_partial`) | 16 / 499 (`483` lost) | 1,855,696.0 | 1,881,391.5 | 1,911,591.0 | 29.5596 | 0.00168 dB |
| Optimized Log-Mel (`optimized_perf.csv`) | `COM5` (3 runs) | 499 / 499 (`0` lost) | **1,711.0** | **1,712.0** | 1,895.0 | **0.0232** | 0.00018 dB |
| Optimized Log-Mel (`optimized_perf.csv`) | `COM10` (3 runs) | 499 / 499 (`0` lost) | **1,705.0** | **1,706.0** | 1,880.0 | **0.0270** | 0.00018 dB |
| Optimized MFCC-13 (`mfcc13_3runs.csv`) | `COM5` (3 runs) | 998 / 998 (`0` lost) | **1,734.0** | **1,739.0** | 1,916.0 | **0.0270** | 0.000092 |

- **Speedup:** **1,084.6×** (`1,855,700 µs` → `1,711 µs`). The median feature computation time (`1.711 ms`) is **8.6% of the `20 ms` (`20,000 µs`) hop-time budget** (not total CPU utilization).

## 3. Six-Class Numerical Accuracy (`six_class_accuracy.csv`)
All 6 Team 02 classes (`on`, `off`, `stop`, `go`, `_unknown`, `_silence`) matched by `frame_numbers` and `first_sample`:
- Max absolute error: **`0.000187 dB`** (`stop`), **`0.000149 dB`** (`go`), **`0.000072 dB`** (`_unknown`), **`0.000053 dB`** (`off`), **`0.000050 dB`** (`on`), **`0.000046 dB`** (`_silence`).
- Mean absolute error: **`0.000003–0.000004 dB`**; RMSE: **`0.000005–0.000009 dB`**. Every frame is well within the `0.05 dB` limit.

## 4. Full 144-Clip Real-Time Log-Mel Stream (`full_144_clips_realtime.csv`)
Streamed all 144 clips on `COM5` (`921600` baud, `16 kHz i16`, `speed=1.0`, `spp=320`):
- `samples_received = 2,304,000 / 2,304,000`, `received_features = 7,199 / 7,199` (`0` lost, `0` `bad_crc`, `0` `seq_gaps`, `0` `uart_errors`).
- `lag_max_s = 0.0234 s`, `compute_median_us = 1,711.0 µs`, `max_abs_err_db = 0.000286 dB`.
- **Clip-boundary frames:** `143` frames cross a 1-second clip boundary (`first_sample // 16000 != (first_sample + 479) // 16000`) and are explicitly flagged rather than assigned a single clip label.

## 5. LLM Usage Log
- Used LLM to implement `mission3_features.py` (including `frame_numbers`/`first_sample` alignment, 6-class validation, 3x MFCC-13, and 144-clip real-time stream) and the Radix-2 FFT in `features.h`. All measurements were executed directly on `COM5` and `COM10`.

## 5. Fastest Sustainable Streaming Speed (`speed_sweep.csv`, `speed_sweep.png`)

**Bottleneck analysis (requested vs. achieved effective speed):**
- **UART theoretical limit:** At `921,600` baud with 10 bits/byte (1 start, 8 data, 1 stop), the raw physical link ceiling is **`92,160 B/s`** before packet header/CRC overhead (13 bytes per 640-byte audio payload).
- **COM5 (921600 baud UART):**
  - `feat=False`: fastest zero-loss effective speed = **2.756×** (requested 16.0×).
  - `feat=True`: fastest zero-loss effective speed = **2.753×** (requested 3.0×, median compute `1752.0` µs/frame).
  - Because COM5 achieves ~2.76× real-time both **with and without** feature computation, **the UART link (`92,160 B/s`) is the primary bottleneck**, not feature compute (`1.71 ms` vs `20 ms` hop budget = `8.6%`).
- **COM10 (Native USB CDC):**
  - `feat=False`: fastest zero-loss effective speed = **3.868×** (requested 4.0×); higher requested speeds overflow the board RX buffer (`bad_crc > 0`, `seq_gaps > 0`).
  - `feat=True`: fastest zero-loss effective speed = **3.873×** (requested 4.0×, median compute `1705.0` µs/frame).

## 6. Per-Class Log-Mel Spectrograms (`per_class_logmel.png`)
Individual log-mel plots generated for one clip from each of the 6 classes using the NumPy reference
pipeline (16 kHz, int16, scale=1.0, antialias=True, win=480, hop=320, nfft=512, mels=40):
- **on** (clip 0), **off** (clip 24), **stop** (clip 48), **go** (clip 72),
  **_unknown** (clip 96), **_silence** (clip 120).

## 7. LLM Usage Update
- Used LLM to implement `mission3_speed_sweep.py` for the speed-sweep (COM5 & COM10, `feat=False/True`) and per-class plot tasks.
