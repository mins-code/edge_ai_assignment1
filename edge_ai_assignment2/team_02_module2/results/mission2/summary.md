# Mission 2: Sampling, Quantisation, and Storage — Results & Analysis (Team 02)

## 1. Quantisation Utilities (`mission2_sampling_storage.py`)
Implemented `quantise()`, `pack()`, and `unpack()` supporting `f32` (`<f4`), `i16` (`<i2`), and `i8` (`int8`) with configurable scale `S`, rounding, symmetric clipping, and clipped-sample counting.

## 2. Sampling & Anti-Aliasing (`aliasing_spectra.png`)
On Team 02 silence clip `120`:
- **48 kHz original spectrum:** Regulator whine peak at **`6,282.0 Hz`**.
- **8 kHz without anti-alias filter (`antialias=False`):** Aliased peak measured at **`1,718.0 Hz`**, matching the theoretical folding formula $|6282 - 8000| = 1,718.0\text{ Hz}$.
- **8 kHz with anti-alias filter (`antialias=True`):** The 255-tap low-pass filter attenuates the `6,282 Hz` tone before decimation, preventing it from folding into the `0–4 kHz` speech band.

## 3. Quantisation Quality (`sqnr_measurements.csv`, `sqnr_comparison.png`)
| Clip | Format | Scale `S` | SQNR (dB) | RMSE | Clipped Samples | 144-Clip Size (16 kHz) |
|---|---|---:|---:|---:|---:|---:|
| Word (`"on"`, `0`) | `i16` | 1.0 | **76.94** | 0.000009 | 0 | 4,608,000 B (4.39 MiB) |
| Word (`"on"`, `0`) | `i8` | 1.0 | 28.67 | 0.002281 | 0 | 2,304,000 B (2.20 MiB) |
| Word (`"on"`, `0`) | `i8` | 0.25 | 20.98 | 0.005528 | 116 | 2,304,000 B |
| Word (`"on"`, `0`) | `i8` | 0.1 | 8.03 | 0.024539 | 1,426 | 2,304,000 B |
| Silence (`120`) | `i16` | 1.0 | **70.68** | 0.000009 | 0 | 4,608,000 B |
| Silence (`120`) | `i8` | 1.0 | 22.41 | 0.002274 | 0 | 2,304,000 B |
| Silence (`120`) | `i8` | 0.25 | 34.42 | 0.000570 | 0 | 2,304,000 B |
| Silence (`120`) | `i8` | 0.1 | 42.48 | 0.000226 | 0 | 2,304,000 B |

**Selected configuration:** `fs = 16000 Hz`, `antialias = True`, `fmt = "i16"`, `scale = 1.0`. It preserves the full `0–8 kHz` speech band and achieves `76.94 dB` SQNR with zero clipping, while fitting in 8 MB PSRAM and 16 MB flash.

## 4. SRAM & PSRAM Capacity (`memory_capacity.csv`)
From `INFO`: `heap_free = 291,528 B` (`heap_largest = 237,556 B`), `psram_free = 8,350,688 B` (`psram_largest = 8,257,524 B`).
- **SRAM (`store="sram"`):** `224 KiB` (`229,376 B`, `114,688` samples = `7.17 s` at `16 kHz i16`) succeeded; `240 KiB` (`245,760 B`) failed (`ERR alloc sram want=245760 largest=237556`).
- **PSRAM (`store="psram"`):** `8,000 KiB` (`8,192,000 B`, `4,096,000` samples = `256.0 s` at `16 kHz i16`) succeeded; `8,192 KiB` (`8,388,608 B`) failed (`ERR alloc psram want=8388608 largest=8257524`).

## 5. Persistent Storage, Custom Partition & 144-Clip Verification
- **Firmware:** Implemented `cmdSave()` in `doorbell_board_starter/doorbell_board_starter.ino` using `LittleFS.open(REC_PATH, "w")` with 4,096-byte chunked writes and `crc32(storeBuf, storeLen)`.
- **Arduino IDE Tools & Partition:** `Board: ESP32S3 Dev Module`, `USB CDC On Boot: Enabled`, `PSRAM: OPI PSRAM`, `Flash Size: 16MB (128Mb)`, `Partition Scheme: Custom` (`doorbell_board_starter/partitions.csv` with a `9.875 MiB` SPIFFS-compatible LittleFS partition → `fs_total = 10,354,688 B`).
- **144-Clip Stream & `SAVE` (`task6_state.json`):** Streamed `4,608,000 B` (`2,304,000` samples) into PSRAM with `0` `bad_crc` and `0` `seq_gaps`, then saved to LittleFS in `10.48 s`.
- **Power-Cycle Persistence (`task7_powercycle.json`):** After unplugging both USB-C cables and reconnecting:
  - `CRC psram`: `n=0, crc=00000000` (PSRAM is volatile; lost on power cut).
  - `CRC flash`: `n=4608000, crc=4c3c7945` (matches original payload `0x4c3c7945`).
  - `DUMP flash`: `4,608,000 B` dumped, **`byte_for_byte_match = True`**.

## 6. LLM Usage Log
- Used LLM to implement `mission2_sampling_storage.py`, `cmdSave()` in `doorbell_board_starter.ino`, and run/verify all hardware experiments on `COM5`.
