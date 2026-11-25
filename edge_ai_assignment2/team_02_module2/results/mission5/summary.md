# Mission 5: Laptop vs Board, Domain Shift & Classification — Results (Team 02)

## 1. Domain Shift Analysis (`domain_shift_metrics.csv`)
Comparing `dataset_board.npz` against `dataset_numpy.npz` (`(144, 49, 40)` on identical `16 kHz i16` audio):
- **Overall:** `max_abs_err = 0.000286 dB`, `mean_abs_err = 0.000003 dB`, `rmse = 0.000006 dB`.
- Implementation-level numerical discrepancy is negligible (`< 0.0003 dB`), whereas real-world acoustic domain shift (e.g., `int8` quantisation, `8 kHz` aliasing without `antialias=True`, or unseen background noise) would cause much larger shifts.

## 2. Nearest-Centroid Cross-Domain Evaluation (`classifier_eval.csv`, `test_predictions.csv`)
- **Representation:** 80-dimensional per-clip `(mean, std)` across the 40 mel bands, standardized using training-set-only `(mu, sd)`.
- **Split:** Stratified 2/3 train (`16` clips/class = `96` clips) and 1/3 test (`8` clips/class = `48` clips), `seed = 42`.

| Experiment | Train / Test | Accuracy | Balanced Acc | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| `NumPy -> NumPy` | 96 / 48 | **79.17%** | 0.7917 | 0.8333 | 0.7917 | 0.7926 |
| `NumPy -> Board` | 96 / 48 | **79.17%** | 0.7917 | 0.8333 | 0.7917 | 0.7926 |
| `Board -> NumPy` | 96 / 48 | **79.17%** | 0.7917 | 0.8333 | 0.7917 | 0.7926 |
| `Board -> Board` | 96 / 48 | **79.17%** | 0.7917 | 0.8333 | 0.7917 | 0.7926 |

- **Cross-domain drop:** **0.00%** — all 48 test clips received identical predictions on both laptop and board features.

## 3. Final Assignment Readiness Audit (Task 9)
- **Completed:**
  - Python scripts: `mission1_pipe.py`, `mission2_sampling_storage.py`, `mission3_features.py`, `mission4_loss_dataset.py`, `mission5_domain_shift.py`.
  - Firmware: `doorbell_board_starter/doorbell_board_starter.ino` (`cmdSave()`), `doorbell_board_starter/features.h` (Radix-2 FFT), `doorbell_board_starter/partitions.csv` (`9.875 MiB` LittleFS).
  - Datasets & Data Card: `results/mission4/dataset_board.npz`, `dataset_numpy.npz`, `data_card.md`.
  - Mission records & LLM logs: `results/mission1..5/summary.md`.
- **Outstanding for final `team_02_module2.zip` submission:**
  - Export/compile the 5 mission records + data card + 1-page client commitment memo + individual team-member role reflections into the final PDF.
  - Package into `team_02_module2/` and verify live demo (`144` clips → `SAVE` → power cycle → `DUMP`/`CRC`).

## 4. LLM Usage Log
- Used LLM to implement `mission5_domain_shift.py`, compute cross-domain nearest-centroid metrics, and compile the final readiness audit.
