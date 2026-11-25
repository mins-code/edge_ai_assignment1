# Project Handoff — Edge AI (Team 02: Edge Experts)

**Repository:** https://github.com/mins-code/edge_ai_assignment1  
**Branch:** `main`  
**Latest Verified Code Commit (Assignment 2):** `fcc1ff8b3df7527542082fd921aa1892216c85d8`  
**Hardware Target (Assignment 2):** ESP32-S3-DevKitC-1 N16R8 (240 MHz, 16 MB Flash, 8 MB OPI PSRAM)  
**Ports:** `COM5` (USB-to-UART bridge, 921,600 baud), `COM10` (Native USB CDC)

---

## 1. Chronological Project Timeline (Assignment 1 → Assignment 2)

### Phase A: Assignment 1 Completion & Repository Restructuring
- **`e255c63` → `d2ff4c5` → `998e7c2`:** Built, trained, evaluated, and served the Team 02 battery model with holdout accuracy verification, clean-clone verification, and an Assignment 1 teammate handoff.
- **`0c8bcf7`:** Moved the repository root to `Edge_AI/` and added the Module 2 assignment folder.
- **`176dbd6`:** Initialized the repository with both `edge_ai_assignment1/` and `edge_ai_assignment2/` modules plus `.devcontainer` setup.
- **`aae6c8f`:** Completed Assignment 1 audit and checks (`edge_ai_assignment1/`).

### Phase B: Assignment 2 ("The Doorbell's Ears") — Missions 1 to 5
- **`99fc6a6` (Mission 1 — The Pipe):** Implemented `mission1_pipe.py` and benchmarked `COM5` (115,200 / 921,600 / 2,000,000 baud) and `COM10` (Native USB) with `samples_per_packet` sweeps (`8, 64, 320, 1024`). Confirmed `COM5` at `921,600` baud with `spp=320` achieves ~88.18 kB/s payload (~89.97 kB/s wire, ~2.76× real-time for 16 kHz `int16`) with 0 errors, while `COM10` overflows RX buffers at unpaced `speed=0`.
- **`646497e` (Mission 2 — Sampling, Quantisation & Persistent Storage):** Implemented `quantise()`, `pack()`, and `unpack()` in `mission2_sampling_storage.py`; added `cmdSave()` (LittleFS persistence) in `doorbell_board_starter/doorbell_board_starter.ino`; created custom `doorbell_board_starter/partitions.csv` (`9.875 MiB` / `10,354,688 B` LittleFS). Selected `16,000 Hz`, `int16`, `scale=1.0`, `antialias=True` (`76.94 dB` word SQNR, `70.68 dB` silence SQNR, `0` clipped samples). Verified saving `4,608,000 B` (144 clips) to LittleFS (`CRC 0x4c3c7945`) and confirmed persistence across a physical USB power cycle.
- **`46f554c` (Mission 3 — Real-Time Feature Extraction & FFT Optimization):** Implemented the NumPy `features()` reference in `mission3_features.py` and replaced the $O(N^2)$ `dft()` in `doorbell_board_starter/features.h` with an in-place Radix-2 Cooley-Tukey FFT using precomputed twiddle-factor and bit-reversal tables. Reduced median compute time from `~1,855,700 µs/frame` to `1,711 µs/frame` (`1,084.6×` speedup, `8.6%` of the `20 ms` hop budget) with `0.000286 dB` max discrepancy on identical quantised `int16` inputs.
- **`b8749d6` (Mission 4 — Loss Accounting & Dataset Construction):** Implemented `mission4_loss_dataset.py` to verify clean 144-clip streaming (`7,200` data packets, `7,201` `frames_ok` incl. EOS, `2,304,000` samples, `7,199` continuous feature frames, `0` errors) and deliberate packet-loss timeline reconstruction (`skip=(10, 11, 200)` and `skip=(10, 49, 150)`). Excluded `143` clip-boundary frames to produce `results/mission4/dataset_board.npz` and `results/mission4/dataset_numpy.npz` of shape `(144, 49, 40)` (`24` clips/class across 6 classes).
- **`a33da44` (Mission 5 — Laptop vs Board, Domain Shift & Classification):** Implemented `mission5_domain_shift.py`, `mission5_preprocessing.py`, and `mission5_board_experiments.py`. Verified `79.17%` nearest-centroid accuracy on both NumPy and Board (`0.00%` cross-domain drop) for `16 kHz int16 S=1.0`, compared against degraded board streams (`int8 S=1.0` → `54.17%`, `int8 S=0.25` → `50.00%`, `8 kHz antialias=False` → `16.67%`).

### Phase C: Audit Remediation, Hardening & Final Verification
- **`5ccd136` → `6bdd2d8` → `07feb52` → `5ee7467`:** Added `mission4_audit.py`, `bad_crc_back` tracking, `first_sample` and `frame_numbers` metadata in both dataset NPZs, `mission5_float32_analysis.py`, `demo_live.py` live demonstration evidence (`results/demo_live_report.json`), `test_mission4_loss_handling.py`, and `package_submission.py`.
- **`15846cd` (Audit Remediation Part 1):**
  - Unified `build_and_validate_datasets()` (`mission4_loss_dataset.py`) and `rebuild_datasets_with_metadata()` (`mission4_audit.py`) through `_shared_dataset_workflow()`, which calls `build_loss_aware_dataset()` and atomic `save_dataset_npz()`.
  - Moved universal integrity checks (`bad_crc`, `bad_crc_back`, `uart_errors`, `NaN`/`Inf`, duplicate frame numbers, `s0 != fn * HOP`) into `_validate_stream_integrity()` so they apply to **both** clean (`skip_tuple=None`) and controlled-loss (`skip_tuple != None`) streams.
  - Updated `results/mission4/summary.md` with effective vs requested speeds from `results/mission4/fastest_speed.csv` (`COM5`: `32.0×` req → `2.737×` eff, `89,360.5 B/s`; `COM10`: `6.0×` req → `5.713×` eff, `186,536.0 B/s`, `feat=False`).
  - Corrected `mission5_float32_analysis.py` and `results/mission5/summary.md` to state that `0.000286 dB` (quantised `int16` NumPy vs Board) satisfies the `0.05 dB` implementation tolerance, whereas `0.246574 dB` (float32 laptop vs Board `int16`) is dominated by `int16` quantisation noise.
  - Upgraded `package_submission.py` to embed 11 experiment PNG figures and formatted observation tables across all 5 missions (8 mandatory headings each) plus the 1-page Client Commitment Memo.
- **`40a33e3` (Starter DFT Hardware Benchmark & Validation Hardening):**
  - Located bundled `arduino-cli.exe` inside the local Arduino IDE installation (`C:\Users\Atul RN\AppData\Local\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe`).
  - Temporarily flashed the original $O(N^2)$ DFT `features.h` from commit `176dbd6` and executed genuine 10-second `speed=1.0` benchmarks on both `COM5` and `COM10` (`16 / 499` frames received, `483` lost, `~1,855,697 µs` median compute, `timeout_partial` at `~30.1 s`, `max_abs_err = 0.00168 dB`). Recorded in `results/mission3/baseline_perf.csv`, `results/mission3/summary.md`, and `results/mission3/baseline_vs_optimized.png`.
  - Restored and re-flashed the optimised Radix-2 FFT `features.h` and verified parity on `COM5` (`99 / 99` frames, `median_us = 1711.0`, `max_abs_err = 0.000172 dB`, `0` errors).
  - Hardened `_validate_stream_integrity()` and `build_loss_aware_dataset()` to validate required metadata keys, `dup == 0`, `packets_sent`, and `frames_ok == expected_packets + 1` (data + EOS); added exact `ci * FS + fi * HOP` offset validation in `mission5_float32_analysis.py`.
- **`fcc1ff8` (Final Polish & PDF Visual Inspection):**
  - Removed fallback defaults for required integrity metadata in both `build_and_validate_datasets()` and `rebuild_datasets_with_metadata()`.
  - Updated `test_20_production_integration_shared_implementation` in `test_mission4_loss_handling.py` to directly invoke both `build_and_validate_datasets()` and `rebuild_datasets_with_metadata()` with mocked `Board` and `tmp_path`.
  - Rendered all 16 pages of `team_02_module2_report.pdf` via PyMuPDF (`1.26.7`), visually inspected every page, fixed bar-label headroom on Page 9 (`Figure 3.1`), rebuilt `team_02_module2/` and `team_02_module2.zip`, and verified all offline tests (`32 passed, 0 failed`).

---

## 2. Current Repository Layout

```text
Edge_AI/
├── HANDOFF.md                          # This project handoff document
├── README.md                           # Root repository overview
├── edge_ai_assignment1/                # Assignment 1 (completed & audited at aae6c8f)
└── edge_ai_assignment2/                # Assignment 2 ("The Doorbell's Ears", Team 02)
    ├── board_link.py                   # Serial/USB binary framing & CRC32 host link
    ├── team_audio.py                   # Deterministic 144-clip generator (unmodified)
    ├── speech_pool.npz                 # Source utterance pool (unmodified)
    ├── demo_live.py                    # End-to-end 144-clip + SAVE + power-cycle + DUMP demo
    ├── mission1_pipe.py                # Mission 1: baud & packet-size sweeps
    ├── mission2_sampling_storage.py    # Mission 2: quantisation, aliasing & SQNR
    ├── mission3_features.py            # Mission 3: NumPy reference & feature benchmarks
    ├── mission3_speed_sweep.py         # Mission 3: speed sweep (feat=False vs feat=True)
    ├── mission4_loss_dataset.py        # Mission 4: fail-closed loss accounting & NPZ builder
    ├── mission4_audit.py               # Mission 4: speed sweep, bad_crc_back & shared rebuild
    ├── test_mission4_loss_handling.py  # 32 offline regression & integration tests
    ├── mission5_domain_shift.py        # Mission 5: NumPy vs Board classifier evaluation
    ├── mission5_preprocessing.py       # Mission 5: offline preprocessing comparisons
    ├── mission5_board_experiments.py   # Mission 5: int8 / 8kHz no-AA board experiments
    ├── mission5_float32_analysis.py    # Mission 5: float32 vs int16 alignment & discrepancy
    ├── package_submission.py           # Generates 16-page PDF, team_02_module2/ & ZIP
    ├── team_02_module2_report.pdf      # Final 16-page submission report (927,328 bytes)
    ├── team_02_module2/                # Packaged submission folder
    ├── team_02_module2.zip             # Verified submission archive (18,215,598 bytes)
    ├── doorbell_board_starter/         # Final optimised ESP32-S3 firmware
    │   ├── doorbell_board_starter.ino  # Sketch with LittleFS SAVE/DUMP/CRC support
    │   ├── features.h                  # Optimised Radix-2 Cooley-Tukey FFT pipeline
    │   └── partitions.csv              # Custom 9.875 MiB LittleFS partition table
    └── results/                        # All verified CSVs, JSONs, PNGs, NPZs & summaries
        ├── demo_live_report.json
        ├── demo_live_state.json
        ├── mission1/
        ├── mission2/
        ├── mission3/
        ├── mission4/
        └── mission5/
```

---

## 3. Verification & Operating Commands

### Run Offline Regression Suite
```powershell
cd edge_ai_assignment2
python -m pytest test_mission4_loss_handling.py -q
```
Expected: `32 passed` (`0` failed).

### Run Offline Float32 Alignment & Discrepancy Analysis
```powershell
cd edge_ai_assignment2
python mission5_float32_analysis.py
```
Expected: `Alignment validation PASSED: 144 clips, 49 frames/clip, 40 mel bands`, `A: max=0.246574 dB`, `B: max=0.000286 dB`, `C: max=0.246574 dB`.

### Rebuild & Verify Submission Package (PDF + ZIP + Extracted Tests)
```powershell
cd edge_ai_assignment2
python package_submission.py
```
Expected: Generates `team_02_module2_report.pdf` (16 pages), `team_02_module2/`, and `team_02_module2.zip`, then runs CRC, SHA-256, NPZ schema, and extracted `pytest` checks.

### Flash Optimised Firmware (Arduino IDE or Bundled `arduino-cli.exe`)
- **Board:** `ESP32S3 Dev Module`
- **USB CDC On Boot:** `Enabled`
- **PSRAM:** `OPI PSRAM`
- **Flash Size:** `16MB (128Mb)`
- **Partition Scheme:** `Custom` (`doorbell_board_starter/partitions.csv`)
- **CLI FQBN:** `esp32:esp32:esp32s3:CDCOnBoot=cdc,PSRAM=opi,FlashSize=16M,PartitionScheme=custom`

### Live Demonstration (`demo_live.py`)
```powershell
cd edge_ai_assignment2
python demo_live.py
```
Streams all 144 clips (`16 kHz`, `int16`, `scale=1.0`, `spp=320`, `feat=True`, `store="psram"`), executes `SAVE` to LittleFS (`CRC 0x4c3c7945`), pauses for physical USB power-cycle, and verifies flash persistence via `CRC` and byte-for-byte `DUMP`.
