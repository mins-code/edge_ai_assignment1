# Edge AI Assignment 2 — Installation Guide
# Team 02 | AI255TBD | The Doorbell's Ears

## Requirements

- Python 3.10 or later (3.11 recommended)
- Arduino IDE 2.x with **esp32 board package ≥ 3.0** installed
- An **ESP32-S3-DevKitC-1 N16R8** development board

---

## 1. Python Environment Setup (Windows)

```powershell
cd c:\Users\Atul RN\Downloads\Edge_AI\edge_ai_assignment2
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Verify Installation

Run the offline regression tests (no hardware required):

```powershell
python -m pytest test_mission4_loss_handling.py -v
```

All 32 tests should pass.

---

## 3. Arduino IDE Setup

### Board Manager

1. Open Arduino IDE → **File → Preferences**
2. Add to Additional Boards Manager URLs:
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
3. Open **Tools → Board → Boards Manager**, search `esp32`, install **esp32 by Espressif Systems ≥ 3.0.0**

### Tools Settings (for ESP32-S3-DevKitC-1 N16R8)

| Setting | Value |
|---|---|
| Board | ESP32S3 Dev Module |
| Upload Speed | 921600 |
| CPU Frequency | 240 MHz |
| Flash Mode | QIO 80 MHz |
| Flash Size | 16 MB |
| Partition Scheme | (Custom) — see `doorbell_board_starter/partitions.csv` |
| PSRAM | OPI PSRAM |
| Arduino Runs On | Core 1 |
| Events Run On | Core 1 |
| USB Mode | Hardware CDC and JTAG |
| USB CDC On Boot | Enabled |
| Upload Mode | UART0 / Hardware CDC |
| Port | COM5 (UART) or COM10 (Native USB) |

### Custom Partition Table

Copy `doorbell_board_starter/partitions.csv` to your Arduino sketch folder and select
**"Custom"** in the Partition Scheme dropdown.

### Firmware Upload

Open `doorbell_board_starter/doorbell_board_starter.ino` in Arduino IDE and click **Upload**.

---

## 4. Live Demonstration

Run the integrated demonstration (requires board on COM5):

```powershell
python demo_live.py --phase stream_save
```

> **Power-cycle required between demo phases:**
> After `stream_save` completes (the `SAVE` command finishes writing to LittleFS), physically
> disconnect and reconnect the USB cable to cut power to the board. Wait at least 3 seconds
> before proceeding. This clears PSRAM and proves that flash storage persists across power loss.

```powershell
python demo_live.py --phase verify
```

Expected output: `PASS — flash CRC matches, all 7199 feature frames received`.

---

## 5. Regenerating Mission Results

| Mission | Script | Hardware? |
|---|---|---|
| M3 speed sweep | `python mission3_speed_sweep.py` | Yes (COM5 + COM10) |
| M4 loss experiments | `python mission4_loss_dataset.py` | Yes (COM5) |
| M4 speed audit | `python mission4_audit.py` | Yes (COM5 + COM10) |
| M5 preprocessing | `python mission5_preprocessing.py` | No |
| M5 float32 analysis | `python mission5_float32_analysis.py` | No |
| M5 board experiments | `python mission5_board_experiments.py` | Yes (COM5) |
| Package submission | `python package_submission.py` | No |

---

## 6. Submission Packaging

```powershell
python package_submission.py
```

Outputs:
- `team_02_module2_report.pdf` — multi-mission report with figures and tables
- `team_02_module2.zip` — full submission archive

The packaging script verifies ZIP integrity, checks all mandatory files, runs `testzip()`,
and validates both NPZ datasets (shape, fields, NaN/Inf).
