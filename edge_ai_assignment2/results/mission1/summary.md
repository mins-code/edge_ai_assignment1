# Mission 1: The Pipe — Results & Analysis (Team 02)

## 1. Theoretical Predictions

- **16 kHz audio data rates:**
  - `float32` (4 B/sample): `64,000 B/s`
  - `int16` (2 B/sample): `32,000 B/s`
  - `int8` (1 B/sample): `16,000 B/s`
- **UART theoretical max (10 bits/byte: 1 start + 8 data + 1 stop):**
  - `115,200` baud: `11,520 B/s` (cannot carry even `16 kHz i8` in real time)
  - `921,600` baud: `92,160 B/s` (can carry `16 kHz f32`, `i16`, `i8`)
  - `2,000,000` baud: `200,000 B/s`

## 2. Port & Baud Throughput (`speed=0`, `16 kHz i16`, `spp=320`)

| Port              |              Baud |  Theoretical (B/s) | Measured (B/s) | `frames_ok` / `sent` | `bad_crc` | `seq_gaps` |
| ----------------- | ----------------: | -----------------: | -------------: | -------------------: | --------: | ---------: |
| `COM5` (UART)     |           115,200 |             11,520 |      11,343.52 |            151 / 150 |         0 |          0 |
| `COM5` (UART)     |           921,600 |             92,160 |      90,369.91 |            151 / 150 |         0 |          0 |
| `COM5` (UART)     |         2,000,000 |            200,000 |     186,067.86 |            151 / 150 |         0 |          0 |
| `COM10` (USB CDC) | 115,200 (ignored) | 1,500,000 (USB FS) |     410,274.82 |             72 / 150 |        31 |          0 |

_(Note: `frames_ok` includes the `'E'` end-of-stream frame, so `151 / 150` indicates 100% of data frames + 1 end frame.)_

## 3. Packet Size & Overhead (`speed=0`, `16 kHz i16`)

Each frame adds **13 bytes** of header + CRC32 (`2` sync + `1` type + `4` seq + `2` len + `4` CRC32).

- `COM5` (921,600 baud): `spp=8` (44.83% overhead) → `51,098.66 B/s`; `spp=64` (9.22%) → `82,177.26 B/s`; `spp=320` (1.99%) → `90,519.45 B/s`; `spp=1024` (0.63%) → `91,932.43 B/s` (all 0 errors).
- `COM10` (USB CDC): `spp=8` → `100,083.41 B/s` (0 errors); `spp=64` → `268,996.55 B/s` (`63` bad CRC, `126` seq gaps); `spp=320` → `426,438.10 B/s` (`28` bad CRC); `spp=1024` → `457,686.63 B/s` (`9` bad CRC).

## 4. Real-Time Feasibility (`speed=1.0`, `spp=320`)

All five formats (`16 kHz f32`, `16 kHz i16`, `16 kHz i8`, `8 kHz i16`, `8 kHz i8`) ran with **0 `bad_crc`, 0 `seq_gaps`, 0 `uart_errors`**, and `frames_ok == packets_sent + 1` on both `COM5` (921,600 baud) and `COM10` (USB CDC).

## 5. Feature Return Bandwidth

With `win=480`, `hop=320` at `16 kHz` (50 frames/s) and `mels=40` (`float32`):

- Raw feature payload per frame: $40 \times 4 = 160\text{ B}$ (+ `8 B` for `first_sample` and `compute_us` = `168 B/frame`)
- Return rate: $50 \times 168 = 8,400\text{ B/s}$ (or $50 \times (168 + 13) = 9,050\text{ B/s}$ including the 13-byte wire frame overhead).

## 6. LLM Usage Log

- Asked LLM for theoretical USB CDC vs. USB-to-UART throughput on ESP32-S3, wire-frame overhead math, and structuring `mission1_pipe.py`. All throughput, packet-loss, and real-time claims were verified directly on `COM5` and `COM10`.
