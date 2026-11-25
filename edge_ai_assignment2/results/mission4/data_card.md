# Data Card — Team 02 Doorbell Audio Feature Dataset

- **Dataset Purpose:** Labelled log-mel spectrogram dataset for spoken-command classification (`on`, `off`, `stop`, `go`, `_unknown`, `_silence`) computed on-device by an ESP32-S3-N16R8.
- **Source & Team:** Synthesized deterministically from `speech_pool.npz` via `team_audio.py` for **Team 02** (`144` one-second clips, `24` clips per class across `6` classes).
- **Audio & Quantisation:** `16,000 Hz`, `antialias=True` (255-tap windowed-sinc low-pass), `int16` (`<i2`), `scale=1.0` (`4,608,000` bytes / `2,304,000` samples total).
- **Feature Extraction:** `win=480` (30 ms), `hop=320` (20 ms), `nfft=512`, `mels=40` (HTK `20–8000 Hz`), `10*log10(max(m, 1e-10))`, computed on ESP32-S3-N16R8 (`240 MHz`, Radix-2 FFT in `features.h`).
- **Transmission Protocol:** `COM5` (`921,600` baud), `320` samples/packet, `speed=1.0`, `0xA5 0x5A` / `0xB5 0x5B` frames with `zlib.crc32`.
- **Loss & Boundary Handling:** `0` `bad_crc`, `0` `seq_gaps`, `0` `uart_errors`, `2,304,000 / 2,304,000` samples, `7,199 / 7,199` frames. `143` frames crossing 1-second clip boundaries (`first_sample // 16000 != (first_sample + 479) // 16000`) were excluded, leaving **`7,056` clean frames** (`49` frames/clip). For loss-affected streams, frames crossing missing-packet gaps or clip boundaries are marked `INVALID` via host-side timeline reconstruction (since aggregate `seq_gaps` alone cannot locate unexpected losses).
- **Dimensions & Files:** `dataset_board.npz` and `dataset_numpy.npz` (`X.shape = (144, 49, 40)` float32, `y.shape = (144,)` int64, `clip_ids.shape = (144,)`).
- **Numerical Validation:** `0` NaNs/Infs; max discrepancy between ESP32-S3 and NumPy reference on identical quantised input is **`0.000286 dB`** (well below `0.05 dB`).
- **Limitations:** Synthetic room noise/hum/whine from `team_audio.py`; `int16` quantisation floor; `first_sample` compresses timeline if packets are lost, so any lossy stream must be reconstructed from `seq` gaps before labelling.
- **Reproducibility:** Run `python mission4_loss_dataset.py` inside `edge_ai_assignment2/` with `COM5` connected.
