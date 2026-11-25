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
