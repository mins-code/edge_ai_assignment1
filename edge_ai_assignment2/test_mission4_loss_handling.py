"""
test_mission4_loss_handling.py -- Deterministic offline regression tests for Mission 4
loss-aware dataset construction and fail-closed unexpected-loss handling.
"""

import numpy as np
import pytest

from mission4_loss_dataset import FS, WIN, HOP, MELS, SPP, CLASSES, build_loss_aware_dataset


def _make_synthetic_stream(n_clips=6, skip_tuple=()):
    """Create a deterministic synthetic stream (offline simulation, not hardware)."""
    n_samples = n_clips * FS
    packets_expected = (n_samples + SPP - 1) // SPP
    orig_map = np.concatenate([
        np.arange(p * SPP, min((p + 1) * SPP, n_samples))
        for p in range(packets_expected) if p not in skip_tuple
    ])
    recv_samples = len(orig_map)
    n_frames = 1 + (recv_samples - WIN) // HOP if recv_samples >= WIN else 0
    fnums = np.arange(n_frames, dtype=np.int64)
    fsamp = fnums * HOP
    # Encode original sample index into feature[0] so we can verify exact alignment
    feats = np.zeros((n_frames, MELS), dtype=np.float32)
    for i, s0 in enumerate(fsamp):
        feats[i, 0] = float(orig_map[s0])
    brd = {
        "samples": recv_samples,
        "bad_crc": 0,
        "seq_gaps": len(skip_tuple),
        "uart_errors": 0,
        "frames_ok": (packets_expected - len(skip_tuple)) + 1,
    }
    expected_total_frames = 1 + (n_samples - WIN) // HOP
    r = {
        "expected_features": expected_total_frames,
        "bad_crc_back": 0,
    }
    clips = [(c * 24, CLASSES[c % len(CLASSES)]) for c in range(n_clips)]
    return feats, fsamp, fnums, brd, r, clips, n_samples


def test_clean_stream():
    feats, fsamp, fnums, brd, r, clips, n_samples = _make_synthetic_stream(6, ())
    res = build_loss_aware_dataset(feats, fsamp, fnums, brd, r, clips, n_samples, skip_tuple=None)
    assert res["X"].shape == (6, 49, MELS)
    assert res["occupied"].all()
    assert res["gap_invalid"] == 0


def test_fail_closed_undisclosed_loss():
    feats, fsamp, fnums, brd, r, clips, n_samples = _make_synthetic_stream(6, (10, 11, 200))
    try:
        build_loss_aware_dataset(feats, fsamp, fnums, brd, r, clips, n_samples, skip_tuple=None)
        assert False, "Expected RuntimeError for undisclosed packet loss"
    except RuntimeError as e:
        assert "Fail-closed" in str(e)


def test_controlled_skip_10_11_200():
    feats, fsamp, fnums, brd, r, clips, n_samples = _make_synthetic_stream(6, (10, 11, 200))
    res = build_loss_aware_dataset(feats, fsamp, fnums, brd, r, clips, n_samples, skip_tuple=(10, 11, 200))
    assert res["gap_invalid"] > 0
    # Verify every valid frame's encoded orig_start matches its assigned clip
    for c_idx in range(6):
        for fic in range(49):
            if res["occupied"][c_idx, fic]:
                orig_s0 = int(res["X"][c_idx, fic, 0])
                assert orig_s0 // FS == c_idx


def test_single_missing_packet_near_boundary():
    feats, fsamp, fnums, brd, r, clips, n_samples = _make_synthetic_stream(6, (49,))
    res = build_loss_aware_dataset(feats, fsamp, fnums, brd, r, clips, n_samples, skip_tuple=(49,))
    assert res["gap_invalid"] >= 1
    assert not res["occupied"][0, 48] or res["gap_invalid"] > 0


def test_duplicate_frame_rejected():
    feats, fsamp, fnums, brd, r, clips, n_samples = _make_synthetic_stream(6, ())
    fnums_dup = fnums.copy()
    fnums_dup[5] = fnums_dup[4]
    try:
        build_loss_aware_dataset(feats, fsamp, fnums_dup, brd, r, clips, n_samples, skip_tuple=())
        assert False, "Expected ValueError for duplicate frame"
    except ValueError as e:
        assert "Duplicate" in str(e)


if __name__ == "__main__":
    test_clean_stream()
    test_fail_closed_undisclosed_loss()
    test_controlled_skip_10_11_200()
    test_single_missing_packet_near_boundary()
    test_duplicate_frame_rejected()
    print("All 5 deterministic offline loss-handling tests PASSED.")
