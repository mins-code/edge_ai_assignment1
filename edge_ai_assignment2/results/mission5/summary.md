# Mission 5: Laptop vs Board, Domain Shift & Classification — Results (Team 02)

## 1. Domain Shift Analysis (`domain_shift_metrics.csv`)
Comparing `dataset_board.npz` against `dataset_numpy.npz` (`(144, 49, 40)` on identical `16 kHz i16` audio):
- **Overall:** `max_abs_err = 0.000286 dB`, `mean_abs_err = 0.000003 dB`, `rmse = 0.000006 dB`.
- Both pipelines receive identical `int16` samples; the tiny discrepancy is pure numerical implementation (single-precision `float` 32-bit FFT in `features.h` vs double-precision `float64` in NumPy).

## 2. Nearest-Centroid Cross-Domain Evaluation (`classifier_eval.csv`, `test_predictions.csv`)
- **Representation:** 80-dimensional per-clip `(mean, std)` across the 40 mel bands, standardized using training-set-only `(mu, sd)`.
- **Split:** Stratified 2/3 train (`16` clips/class = `96` clips) and 1/3 test (`8` clips/class = `48` clips), `seed = 42`.

| Experiment | Train / Test | Accuracy | Balanced Acc | Macro F1 |
|---|---:|---:|---:|---:|
| `NumPy -> NumPy` | 96 / 48 | **79.17%** | 0.7917 | 0.7926 |
| `NumPy -> Board` | 96 / 48 | **79.17%** | 0.7917 | 0.7926 |
| `Board -> NumPy` | 96 / 48 | **79.17%** | 0.7917 | 0.7926 |
| `Board -> Board` | 96 / 48 | **79.17%** | 0.7917 | 0.7926 |

- **Cross-domain drop:** **0.00%** — all 48 test clips received identical predictions on both laptop and board features.

## 3. Actual ESP32 Board Preprocessing Experiments (`board_experiments_*.csv`, `board_experiments_heatmaps.png`)
Streamed all 144 clips on `COM5` (`921600` baud, `speed=1.0`) for each configuration and compared against original laptop reference features (filtered 8 kHz laptop audio for the 8 kHz case):

| Config | Samples | Features | Bad CRC | Seq Gaps | Board vs Quant NumPy Max \|Δ\| | Board vs Laptop Max \|Δ\| | Board vs Laptop RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `int8_S1_16k` | 2,304,000 | 7199 | 0 | 0 | 0.00013 dB | 18.377621 dB | 2.988529 dB |
| `int8_S025_16k` | 2,304,000 | 7199 | 0 | 0 | 0.000088 dB | 37.026131 dB | 3.366306 dB |
| `int16_S1_8k_noaa` | 1,152,000 | 7199 | 0 | 0 | 0.000107 dB | 48.50634 dB | 11.15638 dB |

| Config | Experiment | Accuracy | Balanced Acc |
|---|---|---:|---:|
| `int8_S1_16k` | `Laptop -> Laptop` | 79.17% | 79.17% |
| `int8_S1_16k` | `Laptop -> Board` | 54.17% | 54.17% |
| `int8_S025_16k` | `Laptop -> Laptop` | 79.17% | 79.17% |
| `int8_S025_16k` | `Laptop -> Board` | 50.00% | 50.00% |
| `int16_S1_8k_noaa` | `Laptop -> Laptop` | 68.75% | 68.75% |
| `int16_S1_8k_noaa` | `Laptop -> Board` | 16.67% | 16.67% |

- **int8, S=1.0, 16 kHz:** Step size `1/127 ≈ 0.0079` introduces quantization noise on quiet frames (`RMSE = 2.99 dB` vs float laptop), dropping `Laptop -> Board` accuracy from `79.17%` to `54.17%`.
- **int8, S=0.25, 16 kHz:** Clips all samples with `|x| > 0.25` (`RMSE = 3.37 dB`), dropping `Laptop -> Board` accuracy to `50.00%`.
- **int16, S=1.0, 8 kHz, `antialias=False`:** Severe aliasing folds energy above `4 kHz` into lower mel bands (`RMSE = 11.16 dB` vs filtered 8 kHz laptop), collapsing `Laptop -> Board` accuracy from `68.75%` to `16.67%` (chance).

## 4. Supplementary Offline Cross-Configuration Experiments (`preprocessing_comparison.csv`)
| Config | Accuracy | Balanced Acc |
|---|---:|---:|
| `C1_f32_to_f32` | 79.17% | 79.17% |
| `C1_f32_to_board` | 79.17% | 79.17% |
| `C2_numpy_to_numpy` | 79.17% | 79.17% |
| `C2_numpy_to_board` | 79.17% | 79.17% |
| `C3_i8s1_to_i8s1` | 79.17% | 79.17% |
| `C3_i8s1_to_board` | 56.25% | 56.25% |
| `C4_i8s025_to_i8s025` | 66.67% | 66.67% |
| `C4_i8s025_to_board` | 60.42% | 60.42% |
| `C5a_8k_noaa_to_8k_noaa` | 68.75% | 68.75% |
| `C5b_8k_aa_to_8k_aa` | 68.75% | 68.75% |
| `C5a_8k_noaa_to_8k_aa` | 35.42% | 35.42% |

### Shipping Configuration Justification
**Selected configuration: 16 kHz, `int16`, `S=1.0`, `antialias=True`.** It achieves `76.94 dB` SQNR, `0` clipping, `0.000286 dB` board-vs-NumPy discrepancy, `79.17%` accuracy with **0.00% cross-domain drop**, and `1.711 ms` median compute time (`8.6%` of the `20 ms` hop budget).

## 5. LLM Usage Log
- Used LLM to implement `mission5_domain_shift.py`, `mission5_preprocessing.py`, and `mission5_board_experiments.py`. All board measurements were executed directly on `COM5`.

## Float32-to-Board Discrepancy Analysis

Three offline comparisons using `dataset_board.npz`, `dataset_numpy.npz`, and on-the-fly
float32 feature computation (float64 audio, no quantisation, antialias=True).

**Dataset alignment validation:** Before comparison, explicit checks verified identical
`clip_ids`, `y` (labels), `classes`, `first_sample`, `frame_numbers`, sampling rate (16,000 Hz),
window (480), hop (320), FFT size (512), mel bands (40), 49 frames per clip, and finite values.

| Comparison | Max |Δ| (dB) | Mean |Δ| (dB) | RMSE (dB) | p99 |Δ| (dB) |
|---|---:|---:|---:|---:|
| A: float32 laptop vs board int16 | 0.246574 | 0.006220 | 0.009785 | 0.034024 |
| B: numpy int16 vs board int16    | 0.000286 | 0.000003 | 0.000006 | 0.000023 |
| C: float32 laptop vs numpy int16 | 0.246574 | 0.006220 | 0.009785 | 0.034026 |

**Interpretation:**
- The board-versus-NumPy implementation discrepancy on identical quantised int16 inputs is
  0.000286 dB maximum (comparison B), satisfying the assignment's 0.05 dB
  numerical accuracy requirement.
- The original float32-to-board maximum discrepancy is 0.246574 dB
  (comparison A), primarily caused by quantisation rather than firmware implementation error.
- The 0.05 dB implementation tolerance does not apply to comparisons involving different input
  quantisation. Quantisation errors and implementation errors are not exactly additive in their
  absolute-error statistics: their signed differences combine, but maximum errors and RMSE values
  cannot simply be added.
- Comparison B isolates the board firmware vs NumPy implementation difference,
  confirming the firmware is numerically equivalent on identical inputs.
- Comparison C isolates the float64→int16 quantisation effect.

See `float32_board_heatmap.png` for per-clip, per-band visualisation.
