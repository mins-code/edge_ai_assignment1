"""
mission1_pipe.py -- Mission 1: The Pipe (data rate, format, packet size, and real-time feasibility)
Team 02 | AI255TBD Edge AI Module 2
"""

import csv
import statistics
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from board_link import Board
from team_audio import get_clip

TEAM = 2
COM_PORT = "COM5"
USB_PORT = "COM10"
REPEATS = 3
OUT_DIR = Path("results/mission1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FMT_BYTES = {"f32": 4, "i16": 2, "i8": 1}


def pack_audio(clips_sec: int = 3, fs: int = 16000, fmt: str = "i16") -> bytes:
    """Concatenate `clips_sec` 1-second Team 02 clips and pack into f32, i16, or i8."""
    x = np.concatenate([get_clip(TEAM, i, fs=fs, antialias=True) for i in range(clips_sec)])
    if fmt == "f32":
        return x.astype("<f4").tobytes()
    if fmt == "i16":
        return np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2").tobytes()
    if fmt == "i8":
        return np.clip(np.round(x * 127.0), -128, 127).astype(np.int8).tobytes()
    raise ValueError(f"Unsupported fmt: {fmt}")


def run_stream(port: str, baud: int, payload: bytes, fs: int, fmt: str,
               speed: float, spp: int, set_baud_rate: int | None = None,
               run_idx: int = 1) -> dict:
    """Open `port`, optionally switch UART baud, stream with feat=False, store='none'."""
    bps = FMT_BYTES[fmt]
    samples_expected = len(payload) // bps
    expected_duration_s = samples_expected / fs
    b = Board(port, baud=baud)
    try:
        if set_baud_rate is not None and set_baud_rate != baud:
            b.set_baud(set_baud_rate)
        r = b.stream(
            payload,
            fs=fs,
            fmt=fmt,
            feat=False,
            store="none",
            speed=speed,
            samples_per_packet=spp,
        )
        brd = r["board"]
        elapsed = r["send_seconds"]
        board_ms = brd.get("ms", 0)
        samples_received = brd.get("samples", 0)
        frames_ok = brd.get("frames_ok", 0)
        bad_crc = brd.get("bad_crc", 0)
        seq_gaps = brd.get("seq_gaps", 0)
        uart_errors = brd.get("uart_errors", 0)
        packets_sent = r["packets_sent"]
        bytes_sent = r["bytes_sent"]
        wire_bytes = bytes_sent + packets_sent * 13

        if set_baud_rate is not None and set_baud_rate != baud:
            b.set_baud(baud)

        integrity_ok = (
            bad_crc == 0
            and seq_gaps == 0
            and uart_errors == 0
            and frames_ok == packets_sent + 1
            and samples_received == samples_expected
        )
        return {
            "run": run_idx,
            "port": port,
            "baud": set_baud_rate or baud,
            "fs": fs,
            "fmt": fmt,
            "speed": speed,
            "spp": spp,
            "expected_duration_s": round(expected_duration_s, 3),
            "send_seconds": round(elapsed, 4),
            "board_ms": board_ms,
            "packets_sent": packets_sent,
            "frames_ok": frames_ok,
            "samples_expected": samples_expected,
            "samples_received": samples_received,
            "bytes_sent": bytes_sent,
            "host_payload_bps": round(bytes_sent / elapsed, 2) if elapsed > 0 else 0.0,
            "host_wire_bps": round(wire_bytes / elapsed, 2) if elapsed > 0 else 0.0,
            "board_payload_bps": round((samples_received * bps) / (board_ms / 1000.0), 2) if board_ms > 0 else 0.0,
            "overhead_pct": round(13.0 / (spp * bps + 13.0) * 100.0, 2),
            "bad_crc": bad_crc,
            "seq_gaps": seq_gaps,
            "uart_errors": uart_errors,
            "integrity_ok": integrity_ok,
        }
    finally:
        b.close()


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def aggregate_medians(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        k = tuple(r[gk] for gk in group_keys)
        groups.setdefault(k, []).append(r)
    out = []
    for k, grp in groups.items():
        base = {gk: grp[0][gk] for gk in group_keys}
        base.update({
            "runs": len(grp),
            "median_host_payload_bps": round(statistics.median(r["host_payload_bps"] for r in grp), 2),
            "median_host_wire_bps": round(statistics.median(r["host_wire_bps"] for r in grp), 2),
            "median_board_payload_bps": round(statistics.median(r["board_payload_bps"] for r in grp), 2),
            "median_send_seconds": round(statistics.median(r["send_seconds"] for r in grp), 4),
            "overhead_pct": grp[0]["overhead_pct"],
            "total_bad_crc": sum(r["bad_crc"] for r in grp),
            "total_seq_gaps": sum(r["seq_gaps"] for r in grp),
            "total_uart_errors": sum(r["uart_errors"] for r in grp),
            "all_integrity_ok": all(r["integrity_ok"] for r in grp),
        })
        out.append(base)
    return out


def main():
    payload_3s_i16 = pack_audio(clips_sec=3, fs=16000, fmt="i16")

    # 1. Baud-rate & port throughput sweep (speed=0, 3 repeats)
    print("=== 1. Port & Baud Throughput Sweep (speed=0, 16kHz i16, spp=320, 3 runs) ===")
    baud_raw = []
    for baud in (115200, 921600, 2000000):
        for run_idx in range(1, REPEATS + 1):
            row = run_stream(COM_PORT, 921600, payload_3s_i16, 16000, "i16", 0.0, 320, set_baud_rate=baud, run_idx=run_idx)
            row["theoretical_bps"] = baud // 10
            baud_raw.append(row)
    for run_idx in range(1, REPEATS + 1):
        row = run_stream(USB_PORT, 115200, payload_3s_i16, 16000, "i16", 0.0, 320, run_idx=run_idx)
        row["theoretical_bps"] = 1_500_000
        baud_raw.append(row)
    baud_med = aggregate_medians(baud_raw, ("port", "baud", "theoretical_bps"))
    for r in baud_med:
        print(r)
    save_csv(OUT_DIR / "baud_sweep_raw.csv", baud_raw)
    save_csv(OUT_DIR / "baud_sweep.csv", baud_med)

    # 2. Packet size sweep (speed=0, 3 repeats)
    print("\n=== 2. Packet Size Sweep (speed=0, 16kHz i16, 3 runs) ===")
    pkt_raw = []
    for port, baud in [(COM_PORT, 921600), (USB_PORT, 115200)]:
        for spp in (8, 64, 320, 1024):
            for run_idx in range(1, REPEATS + 1):
                row = run_stream(port, baud, payload_3s_i16, 16000, "i16", 0.0, spp, run_idx=run_idx)
                pkt_raw.append(row)
    pkt_med = aggregate_medians(pkt_raw, ("port", "baud", "spp"))
    for r in pkt_med:
        print(r)
    save_csv(OUT_DIR / "packet_size_sweep_raw.csv", pkt_raw)
    save_csv(OUT_DIR / "packet_size_sweep.csv", pkt_med)

    # 3. Real-time feasibility (speed=1.0, 3 repeats)
    print("\n=== 3. Real-Time Feasibility (speed=1.0, spp=320, 3 runs) ===")
    rt_raw = []
    configs = [(16000, "f32"), (16000, "i16"), (16000, "i8"), (8000, "i16"), (8000, "i8")]
    for port, baud in [(COM_PORT, 921600), (USB_PORT, 115200)]:
        for fs, fmt in configs:
            pl = pack_audio(clips_sec=2, fs=fs, fmt=fmt)
            for run_idx in range(1, REPEATS + 1):
                row = run_stream(port, baud, pl, fs, fmt, 1.0, 320, run_idx=run_idx)
                row["required_bps"] = fs * FMT_BYTES[fmt]
                row["realtime_ok"] = row["integrity_ok"]
                rt_raw.append(row)
    rt_med = aggregate_medians(rt_raw, ("port", "baud", "fs", "fmt", "required_bps"))
    for r in rt_med:
        r["realtime_ok"] = r["all_integrity_ok"]
        print(r)
    save_csv(OUT_DIR / "realtime_feasibility_raw.csv", rt_raw)
    save_csv(OUT_DIR / "realtime_feasibility.csv", rt_med)

    # 4. Improved Plots
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    uart_m = [r for r in baud_med if r["port"] == COM_PORT]
    ax.plot([r["baud"] for r in uart_m], [r["median_board_payload_bps"] / 1000 for r in uart_m],
            "o-", color="tab:blue", label="COM5 Board Payload (reliable)")
    ax.plot([r["baud"] for r in uart_m], [r["median_host_wire_bps"] / 1000 for r in uart_m],
            "s--", color="tab:cyan", label="COM5 Host Wire (incl. 13B/pkt)")
    ax.plot([r["baud"] for r in uart_m], [r["theoretical_bps"] / 1000 for r in uart_m],
            ":", color="black", label="UART Theoretical (baud/10)")
    usb_m = [r for r in baud_med if r["port"] == USB_PORT][0]
    usb_style = "o" if usb_m["all_integrity_ok"] else "X"
    usb_lbl = f"COM10 USB Board Payload ({usb_m['median_board_payload_bps']/1000:.1f} kB/s, {'OK' if usb_m['all_integrity_ok'] else 'CORRUPTED'})"
    ax.axhline(usb_m["median_board_payload_bps"] / 1000, color="tab:red" if not usb_m["all_integrity_ok"] else "tab:green",
               linestyle="-.", label=usb_lbl)
    ax.set_xlabel("UART Baud Rate (bps)")
    ax.set_ylabel("Throughput (kB/s)")
    ax.set_title("Mission 1: Median Throughput vs Baud Rate (3 runs, spp=320)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "throughput_vs_baud.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for port, label, color in [(COM_PORT, "COM5 (921600)", "tab:blue"), (USB_PORT, "COM10 (USB)", "tab:orange")]:
        sub = [r for r in pkt_med if r["port"] == port]
        xs = [r["spp"] for r in sub]
        ys_board = [r["median_board_payload_bps"] / 1000 for r in sub]
        ys_wire = [r["median_host_wire_bps"] / 1000 for r in sub]
        ax.plot(xs, ys_board, "-", color=color, label=f"{label} Board Payload")
        ax.plot(xs, ys_wire, "--", color=color, alpha=0.6, label=f"{label} Host Wire")
        for r in sub:
            marker = "o" if r["all_integrity_ok"] else "X"
            mcol = color if r["all_integrity_ok"] else "tab:red"
            ax.scatter([r["spp"]], [r["median_board_payload_bps"] / 1000], marker=marker, color=mcol, s=60, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 64, 320, 1024])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Samples per Packet")
    ax.set_ylabel("Throughput (kB/s)")
    ax.set_title("Mission 1: Packet Size vs Throughput (o = 0 errors, X = CRC/seq loss)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "packet_size_overhead.png", dpi=150)
    plt.close(fig)

    print("\nSaved all raw/median CSVs and plots to", OUT_DIR)


if __name__ == "__main__":
    main()
