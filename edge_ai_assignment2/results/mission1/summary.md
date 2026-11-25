# Mission 1: The Pipe — Results & Analysis (Team 02)

## 1. Theoretical Predictions

- **16 kHz audio data rates:** `f32` = `64,000 B/s`, `i16` = `32,000 B/s`, `i8` = `16,000 B/s` (`8 kHz i16` = `16,000 B/s`, `8 kHz i8` = `8,000 B/s`).
- **UART theoretical max (10 bits/byte: 1 start + 8 data + 1 stop):** `115,200` baud = `11,520 B/s`, `921,600` baud = `92,160 B/s`, `2,000,000` baud = `200,000 B/s`.

## 2. Port & Baud Throughput (`speed=0`, `16 kHz i16`, `spp=320`, 3 runs)

| Port              |              Baud | Theoretical (B/s) | Median Host Payload (B/s) | Median Host Wire (B/s) | Median Board Payload (B/s) | `bad_crc` / `seq_gaps` | Integrity OK |
| ----------------- | ----------------: | ----------------: | ------------------------: | ---------------------: | -------------------------: | ---------------------: | :----------: |
| `COM5` (UART)     |           115,200 |            11,520 |                 11,357.76 |              11,588.47 |                  11,369.02 |                  0 / 0 |     True     |
| `COM5` (UART)     |           921,600 |            92,160 |                 89,611.28 |              91,431.51 |                  89,719.63 |                  0 / 0 |     True     |
| `COM5` (UART)     |         2,000,000 |           200,000 |                184,745.42 |             188,498.06 |                 184,615.38 |                  0 / 0 |     True     |
| `COM10` (USB CDC) | 115,200 (ignored) |         1,500,000 |                379,418.75 |             387,125.69 |                 147,415.73 |              100 / 145 |  **False**   |

## 3. Packet Size & Overhead (`speed=0`, `16 kHz i16`, 3 runs)

Each frame adds **13 bytes** of header + CRC32.

- **`COM5` (921,600 baud, all 0 errors):** `spp=8` (44.83% overhead) → `51,036.68 B/s` board payload (`92,462.41 B/s` wire); `spp=64` (9.22%) → `83,333.33 B/s`; `spp=320` (1.99%) → `89,971.88 B/s`; `spp=1024` (0.63%) → `91,954.02 B/s`.
- **`COM10` (USB CDC, uncapped `speed=0`):** `spp=8` → `86,769.23 B/s` board (`1,167` bad CRC, `2,952` seq gaps); `spp=64` → `167,429.89 B/s` (`206` bad CRC, `404` seq gaps); `spp=320` → `179,414.23 B/s` (`87` bad CRC, `81` seq gaps); `spp=1024` → `138,847.46 B/s` (`28` bad CRC, `56` seq gaps).

## 4. Real-Time Feasibility (`speed=1.0`, `spp=320`, 3 runs, 2 s clips)

All 5 formats (`16 kHz f32`, `16 kHz i16`, `16 kHz i8`, `8 kHz i16`, `8 kHz i8`) achieved **`realtime_ok = True`** on both `COM5` (921,600 baud) and `COM10` (USB CDC), with `0` `bad_crc`, `0` `seq_gaps`, `0` `uart_errors`, `frames_ok == packets_sent + 1`, and `samples_received == samples_expected`.

## 5. Feature Return Bandwidth

With `win=480`, `hop=320` at `16 kHz` (`50 frames/s`) and `mels=40` (`float32`):

- `40 * 4 + 8 = 168 B/frame` → **`8,400 B/s`** payload, or **`9,050 B/s`** including the 13-byte wire frame overhead.
