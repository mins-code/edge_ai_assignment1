# Mission 4: Loss Accounting & Dataset Construction — Results (Team 02)

## 1. Deliberate Packet-Loss Experiments (`loss_experiments.csv`, `loss_comparison.png`)
| Experiment | `skip` | Sent / Exp Packets | `seq_gaps` | `bad_crc` | Recv / Exp Samples | Recv / Exp Features | Lost Features |
|---|---|---:|---:|---:|---:|---:|---:|
| `control_no_loss` | `()` | 300 / 300 | 0 | 0 | 96,000 / 96,000 | 299 / 299 | 0 |
| `readme_skip_10_11_200` | `(10, 11, 200)` | 297 / 300 | **3** | 0 | 95,040 / 96,000 | 296 / 299 | **3** |
| `single_beginning_5` | `(5,)` | 299 / 300 | **1** | 0 | 95,680 / 96,000 | 298 / 299 | **1** |
| `burst_20_to_23` | `(20, 21, 22, 23)` | 296 / 300 | **4** | 0 | 94,720 / 96,000 | 295 / 299 | **4** |

- **Timeline behavior:** `doorbell_board_starter.ino` increments `st.samples` only on received samples, so `first_sample` compresses the timeline by `len(skipped) * 320` samples after a gap. Aggregate `seq_gaps` alone cannot locate unexpected losses; for controlled experiments, host-side reconstruction using known skipped sequence numbers maps received sample indices back to original sample positions.
- **Multi-class loss-aware alignment (`multiclass_loss_frames.csv`):** Streamed 6 clips across all 6 classes (`on`, `off`, `stop`, `go`, `_unknown`, `_silence`) with `skip=(10, 49, 150)` (`49` near the `on`→`off` boundary). Reconstructed timeline (`timeline_shift_samples = -960`): `296` received frames → `3` gap-crossing frames invalidated, `3` boundary-crossing frames invalidated, **`290` valid frames** matched original NumPy reference frames with **`max_valid_err_db = 0.000187 dB`**.

## 2. Clean 144-Clip Dataset (`dataset_board.npz`, `dataset_numpy.npz`, `dataset_validation.json`)
- Streamed all 144 clips on `COM5` (`921600` baud, `16 kHz i16`, `speed=1.0`, `spp=320`): `2,304,000 / 2,304,000` samples, `7,199 / 7,199` frames, `0` `bad_crc`, `0` `seq_gaps`, `0` `uart_errors`.
- Excluded `143` boundary-crossing frames → **`X.shape = (144, 49, 40)`**, **`y.shape = (144,)`**, `24` clips per class across all `6` classes, `0` NaNs/Infs, max board-vs-NumPy discrepancy = **`0.000286 dB`**.

## 3. LLM Usage Log
- Used LLM to implement `mission4_loss_dataset.py`, run the deliberate packet-loss and 144-clip dataset experiments on `COM5`, and draft `data_card.md`.

## 3. Fastest Zero-Loss Streaming Speed (`fastest_speed.csv`)

Swept `speed` from 0.5× to 32× on both USB interfaces with `feat=False, store="none"` (requiring `bad_crc==0`, `seq_gaps==0`, `uart_errors==0`, `bad_crc_back==0`, `samples_received==samples_expected`, and `packets_sent==packets_expected`):
- **COM5 (921600 baud, UART):** Fastest zero-loss requested speed = **32.0×**
- **COM10 (native USB, 115200 baud):** Fastest zero-loss requested speed = **6.0×**
- `bad_crc_back` (return-path CRC failures) = 0 at all zero-loss speeds.

## 4. Full 144-Clip Loss Accounting (Verified)

| Metric | Clean 144-Clip | Full 144-Clip `skip=(10,11,200)` |
|---|---:|---:|
| Clips | 144 | 144 |
| Packets expected | 7,200 | 7,200 |
| Packets sent | 7200 | 7197 |
| `board["frames_ok"]` | 7201 *(incl. EOS)* | 7198 *(incl. EOS)* |
| Samples expected | 2,304,000 | 2,304,000 |
| Samples received | 2,304,000 | 2,303,040 |
| Expected feature frames | 7,199 | 7,199 |
| Received feature frames | 7199 | 7196 |
| Lost feature frames | 0 | 3 |
| `bad_crc` | 0 | 0 |
| `bad_crc_back` | 0 | 0 |
| `seq_gaps` | 0 | 3 |
| `uart_errors` | 0 | 0 |
| Boundary-crossing frames excluded | 143 | — |
| Clean within-clip frames in dataset | 7,056 (144 × 49) | — |

## 5. Dataset Metadata Update (`dataset_board.npz`, `dataset_numpy.npz`)
Both NPZ files now contain additional arrays:
- `first_sample`: shape `(144, 49)` — original `first_sample` offset for each stored feature frame.
- `frame_numbers`: shape `(144, 49)` — global frame index for each stored feature frame.
- `X`, `y`, `classes`, `clip_ids` are **bit-identical** to the previous dataset build.

## 6. LLM Usage Update
- Used LLM to implement `mission4_audit.py` for the speed sweep, `bad_crc_back` enrichment,
  full-stream `skip=(10,11,200)` experiment, dataset metadata addition, and 144-clip loss accounting verification.
