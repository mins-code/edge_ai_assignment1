# Data Card — Team 02 Doorbell Audio Feature Dataset

## 1. Dataset Source & Team-Specific Audio Generation
- **Source:** Synthesized deterministically from `speech_pool.npz` (780 recorded utterances) via `team_audio.py` for **Team 02 (Edge experts)**.
- **Reproducibility:** `team_audio.get_clip(2, cid, fs=16000, antialias=True)` deterministically generates each 1-second clip.

## 2. Classes, Clip Counts & Class Balance
- **Six classes:** `['on', 'off', 'stop', 'go', '_unknown', '_silence']`.
- **Clip count:** **144** one-second clips total, **24 clips per class** (perfectly balanced).

## 3. Sampling & Quantisation Settings
- **Sampling rate:** `16,000 Hz`, `antialias=True` (255-tap windowed-sinc low-pass filter).
- **Format & Scale:** `int16` (`<i2`), `scale=1.0` (`2,304,000` samples = `4,608,000` bytes).
- **SQNR:** `76.94 dB` (word) / `70.68 dB` (silence), `0` clipped samples.

## 4. Feature Extraction Parameters
- **Window:** `win=480` samples (`30 ms`), Hann window.
- **Hop:** `hop=320` samples (`20 ms`, `50 frames/s`).
- **FFT & Mel Filterbank:** `nfft=512`, `mels=40` HTK mel triangles (`20 Hz` to `8,000 Hz`), `10 * log10(max(m, 1e-10))`.
- **On-board implementation:** ESP32-S3-N16R8 (`240 MHz`), in-place Radix-2 Cooley-Tukey FFT in `doorbell_board_starter/features.h` (`1,711 µs` median compute/frame).

## 5. Transport Configuration
- **Port & Baud:** `COM5` (USB-to-UART bridge) at `921,600` baud, `samples_per_packet=320`, `speed=1.0`.
- **Framing:** `0xA5 0x5A` (uplink) / `0xB5 0x5B` (downlink) with `zlib.crc32`.

## 6. Loss Accounting & Boundary-Frame Exclusion
- **Clean 144-clip stream:** `7,200` data packets sent, `7,201` `frames_ok` (incl. EOS), `2,304,000 / 2,304,000` samples received, `7,199 / 7,199` continuous feature frames received, `0` `bad_crc`, `0` `bad_crc_back`, `0` `seq_gaps`, `0` `uart_errors`.
- **Boundary-frame exclusion:** `143` frames crossing 1-second clip boundaries (`first_sample // 16000 != (first_sample + 479) // 16000`) were excluded, leaving **`7,056` clean within-clip frames** (`49` frames/clip).

## 7. Unexpected Packet-Loss Handling
- `doorbell_board_starter.ino` increments `st.samples` only on received samples, compressing `first_sample` after gaps.
- `build_loss_aware_dataset()` in `mission4_loss_dataset.py` enforces **fail-closed** behaviour: any stream with unexplained `bad_crc`, `seq_gaps`, `uart_errors`, `bad_crc_back`, or sample/feature count mismatch raises `RuntimeError` rather than silently labelling compressed-timeline features.
- For controlled experiments with explicitly known `skip_tuple`, host-side `orig_map` reconstruction invalidates gap-crossing and boundary-crossing frames.

## 8. Dataset Files, Fields, Dimensions & Data Types
Both `results/mission4/dataset_board.npz` (ESP32-S3 features) and `results/mission4/dataset_numpy.npz` (matched NumPy reference) contain:
- `X`: `float32`, shape **`(144, 49, 40)`** — log-mel features (dB).
- `y`: `int64`, shape **`(144,)`** — class index (`0..5`).
- `clip_ids`: `int64`, shape **`(144,)`** — Team 02 clip IDs (`0..143`).
- `classes`: `<U8`, shape **`(6,)`** — class names.
- `first_sample`: `int64`, shape **`(144, 49)`** — original sample offset per frame.
- `frame_numbers`: `int64`, shape **`(144, 49)`** — global frame index per frame.
- Scalar metadata: `fs=16000`, `win=480`, `hop=320`, `nfft=512`, `mels=40`, `fmt='i16'`, `scale=1.0`.

## 9. Dataset Validation
- Verified `0` NaNs/Infs, `occupied.all() == True` (no unfilled zero slots), exact `24` clips/class, and max board-vs-NumPy discrepancy = **`0.000286 dB`** (`< 0.05 dB` limit).

## 10. Reproduction Instructions
```bash
python mission4_loss_dataset.py
python mission4_audit.py
```

## 11. Known Limitations & Appropriate / Inappropriate Use Cases
- **Limitations:** Synthetic room noise/hum/whine from `team_audio.py`; `int16` quantisation floor; single-speaker-pool synthesis.
- **Appropriate use:** Benchmarking on-device feature extraction, numerical parity, and pipeline loss accounting.
- **Inappropriate use:** This dataset is **NOT** evidence of production-grade speech recognition, real-world microphone robustness, speaker authentication, or generalisation to arbitrary environmental conditions.
