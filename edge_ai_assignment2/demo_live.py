"""
demo_live.py -- Integrated live demonstration script for Assignment 2: The Doorbell's Ears
Team 02 | AI255TBD Edge AI Module 2

This script performs the full end-to-end demonstration:
  1. Stream all 144 clips with feat=True AND store="psram" simultaneously.
  2. Validate returned features and sample counts (zero errors required).
  3. Execute SAVE to LittleFS.
  4. Instruct user to power-cycle (disconnect both USB cables).
  5. After reconnect: verify CRC and byte-for-byte DUMP match.

Usage:
  python demo_live.py

COM5 must be connected and the ESP32-S3 must have the optimized Radix-2 FFT
firmware loaded with the custom partitions.csv (9.875 MiB LittleFS).
"""

import sys
import time
import zlib
from pathlib import Path

import numpy as np

from board_link import Board
from team_audio import CLASSES, clip_list, get_clip
from mission2_sampling_storage import quantise, pack
from mission3_features import features as compute_features

# ---- Configuration ----
COM_PORT = "COM5"
BAUD = 921600
FS = 16000
FMT = "i16"
SCALE = 1.0
WIN = 480
HOP = 320
NFFT = 512
MELS = 40
SPP = 320
TEAM = 2


def hr(char="=", n=70):
    print(char * n)


def step(n, text):
    hr()
    print(f"  STEP {n}: {text}")
    hr()


def demo():
    hr()
    print("  Team 02 — The Doorbell's Ears: Integrated Live Demonstration")
    print("  ESP32-S3-N16R8 | 240 MHz | 16 MB Flash | 8 MB PSRAM")
    print("  COM5 (UART 921600 baud) | 16 kHz i16 | Radix-2 FFT features")
    hr()

    # ------------------------------------------------------------------ Step 1
    step(1, "Connect to board and verify configuration")
    b = Board(COM_PORT, baud=BAUD)
    try:
        info = b.info()
        print(f"  INFO: {info}")
        fs_total = info.get("fs_total", 0)
        fs_used  = info.get("fs_used", 0)
        print(f"  LittleFS: {fs_total:,} B total, {fs_used:,} B used, {fs_total - fs_used:,} B free")
        assert fs_total >= 9_000_000, f"Unexpected LittleFS size: {fs_total}"
    finally:
        b.close()

    # ------------------------------------------------------------------ Step 2
    step(2, "Prepare all 144 clips for streaming")
    clips = clip_list(TEAM)
    print(f"  Loading {len(clips)} clips ...")
    x_all = np.concatenate([get_clip(TEAM, cid, fs=FS, antialias=True) for cid, _ in clips])
    _, xhat_all, _ = quantise(x_all, fmt=FMT, scale=SCALE)
    pl_all = pack(x_all, fmt=FMT, scale=SCALE)
    n_samples_expected = len(x_all)
    n_frames_expected = 1 + (n_samples_expected - WIN) // HOP if n_samples_expected >= WIN else 0
    n_bytes_expected = n_samples_expected * 2  # int16

    print(f"  Clips: {len(clips)}, Samples: {n_samples_expected:,}, "
          f"Payload: {len(pl_all):,} B, Expected frames: {n_frames_expected}")

    # Build NumPy reference for validation
    print("  Computing NumPy reference features ...")
    ref_all = compute_features(xhat_all, fs=FS, win=WIN, hop=HOP, nfft=NFFT, mels=MELS)
    print(f"  Reference frames: {len(ref_all)}")

    # ------------------------------------------------------------------ Step 3
    step(3, "Stream all 144 clips with feat=True AND store='psram'")
    b = Board(COM_PORT, baud=BAUD)
    stream_result = None
    try:
        print("  Streaming ...")
        t0 = time.perf_counter()
        r = b.stream(pl_all, fs=FS, fmt=FMT, scale=SCALE, win=WIN, hop=HOP,
                     nfft=NFFT, mels=MELS, mfcc=0, feat=True, store="psram",
                     speed=1.0, samples_per_packet=SPP)
        elapsed = time.perf_counter() - t0
        stream_result = r
        print(f"  Stream complete in {elapsed:.1f}s")

        brd = r["board"] or {}
        n_recv_samples = brd.get("samples", 0)
        n_recv_frames  = len(r["features"])
        bad_crc        = brd.get("bad_crc", 0)
        seq_gaps       = brd.get("seq_gaps", 0)
        uart_err       = brd.get("uart_errors", 0)
        bad_crc_back   = r.get("bad_crc_back", 0)
        lost_features  = r["lost_features"]
        lag_max        = float(np.max(r["lag_s"])) if len(r["lag_s"]) else 0.0
        compute_med    = float(np.median(r["compute_us"])) if len(r["compute_us"]) else 0.0

        print(f"  Samples received:  {n_recv_samples:,} / {n_samples_expected:,}")
        print(f"  Features received: {n_recv_frames} / {n_frames_expected}")
        print(f"  Lost features:     {lost_features}")
        print(f"  bad_crc:           {bad_crc}")
        print(f"  seq_gaps:          {seq_gaps}")
        print(f"  uart_errors:       {uart_err}")
        print(f"  bad_crc_back:      {bad_crc_back}")
        print(f"  Lag max:           {lag_max:.4f}s")
        print(f"  Compute median:    {compute_med:.1f} µs/frame")

        # Validate feature accuracy
        fnums = np.asarray(r["frame_numbers"], dtype=np.int64)
        got_feat = r["features"]
        err_all = np.abs(got_feat - ref_all[fnums])
        max_err = float(np.max(err_all)) if len(err_all) else float("nan")
        print(f"  Max feature error: {max_err:.6f} dB (limit: 0.05 dB)")

        # Assertions
        assert n_recv_samples == n_samples_expected, \
            f"Sample count mismatch: {n_recv_samples} != {n_samples_expected}"
        assert n_recv_frames == n_frames_expected, \
            f"Frame count mismatch: {n_recv_frames} != {n_frames_expected}"
        assert bad_crc == 0 and seq_gaps == 0 and uart_err == 0 and bad_crc_back == 0, \
            f"Errors detected: bad_crc={bad_crc} seq_gaps={seq_gaps} uart_err={uart_err} bcb={bad_crc_back}"
        assert lost_features == 0, f"Lost features: {lost_features}"
        assert max_err <= 0.05, f"Feature error too large: {max_err:.6f} dB"
        print("\n  ✓ STEP 3 PASSED: All 144 clips streamed, features validated, PSRAM populated.")

    finally:
        b.close()

    # ------------------------------------------------------------------ Step 4
    step(4, "SAVE to LittleFS flash")
    b = Board(COM_PORT, baud=BAUD)
    save_info = None
    try:
        print("  Sending SAVE command ...")
        save_info = b.save()
        print(f"  SAVE result: {save_info}")
        n_saved = save_info.get("n", 0)
        crc_saved = save_info.get("crc", 0)
        print(f"  Saved {n_saved:,} bytes to flash, CRC = 0x{crc_saved:08x}")
        assert n_saved == n_bytes_expected, \
            f"Saved bytes mismatch: {n_saved} != {n_bytes_expected}"
        print(f"\n  ✓ STEP 4 PASSED: {n_saved:,} bytes saved to LittleFS. CRC = 0x{crc_saved:08x}")
    finally:
        b.close()

    crc_expected = save_info.get("crc", 0) if save_info else 0

    # ------------------------------------------------------------------ Step 5
    step(5, "Power-cycle verification")
    print("\n  *** ACTION REQUIRED ***")
    print("  1. DISCONNECT both USB cables from the ESP32-S3.")
    print("  2. Wait at least 2 seconds (until PSRAM is fully cleared).")
    print("  3. RECONNECT both USB cables.")
    print("  4. Wait for the board to boot (the native USB COM10 may take ~5 seconds).")
    print()
    input("  Press ENTER when the board is reconnected and ready ...")
    time.sleep(2.0)

    # ------------------------------------------------------------------ Step 6
    step(6, "Verify PSRAM is empty and flash CRC matches")
    b = Board(COM_PORT, baud=BAUD)
    try:
        # PSRAM should be zeroed after power cycle
        psram_crc = b.crc("psram")
        print(f"  CRC psram: {psram_crc}  (n should be 0 — PSRAM cleared by power cycle)")
        assert psram_crc.get("n", -1) == 0, \
            f"PSRAM not empty after power cycle: n={psram_crc.get('n')}"

        # Flash CRC should match what we saved
        flash_crc = b.crc("flash")
        print(f"  CRC flash: {flash_crc}  (crc should be 0x{crc_expected:08x})")
        n_flash = flash_crc.get("n", 0)
        crc_flash = flash_crc.get("crc", 0)
        assert n_flash == n_bytes_expected, \
            f"Flash byte count mismatch: {n_flash} != {n_bytes_expected}"
        assert crc_flash == crc_expected, \
            f"Flash CRC mismatch: 0x{crc_flash:08x} != 0x{crc_expected:08x}"
        print(f"\n  ✓ STEP 6 PASSED: PSRAM cleared, flash CRC 0x{crc_flash:08x} matches saved data.")

        # ---------------------------------------------------------------- Step 7
        step(7, "Byte-for-byte DUMP verification")
        print("  Dumping flash contents ...")
        dump_bytes = b.dump("flash")
        print(f"  Dumped {len(dump_bytes):,} bytes")
        assert len(dump_bytes) == n_bytes_expected, \
            f"Dump length mismatch: {len(dump_bytes)} != {n_bytes_expected}"

        # Verify byte-for-byte against original payload
        assert dump_bytes == pl_all[:n_bytes_expected], \
            "Byte-for-byte mismatch between flash dump and original payload!"
        dump_crc = zlib.crc32(dump_bytes) & 0xFFFFFFFF
        print(f"  Dump CRC (zlib): 0x{dump_crc:08x}")
        print(f"\n  ✓ STEP 7 PASSED: Flash dump is byte-for-byte identical to original payload.")

    finally:
        b.close()

    # ------------------------------------------------------------------ Summary
    hr("*")
    print("  INTEGRATED DEMO COMPLETE — ALL STEPS PASSED")
    hr("-")
    print(f"  Clips streamed:        {len(clips)}")
    print(f"  Samples:               {n_samples_expected:,}")
    print(f"  Feature frames:        {n_frames_expected}")
    print(f"  PSRAM stored:          {n_bytes_expected:,} bytes")
    print(f"  Flash CRC:             0x{crc_expected:08x}")
    print(f"  Power-cycle verified:  YES (PSRAM cleared, flash persisted)")
    print(f"  Byte-for-byte check:   PASS")
    hr("*")


if __name__ == "__main__":
    demo()
