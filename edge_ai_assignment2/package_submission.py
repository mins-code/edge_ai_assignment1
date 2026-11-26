"""
package_submission.py -- Generate final 8-heading PDF report with embedded figures and tables,
                         clean pycache, build team_02_module2/ and team_02_module2.zip, and verify.
Team 02 | AI255TBD Edge AI Module 2
"""

import csv
import hashlib
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(".")
PKG = ROOT / "team_02_module2"
ZIP_PATH = ROOT / "team_02_module2.zip"
PDF_PATH = ROOT / "team_02_module2_report.pdf"


def add_text_page(pdf: PdfPages, title: str, body: str, fontsize: float = 8.2):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.95, title, fontsize=12.5, fontweight="bold", va="top")
    wrapped_lines = []
    for line in body.splitlines():
        if not line.strip():
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=102, replace_whitespace=False))
    lines_per_page = 66
    for start in range(0, max(1, len(wrapped_lines)), lines_per_page):
        if start > 0:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.06, 0.95, f"{title} (cont.)", fontsize=11, fontweight="bold", va="top")
        chunk = "\n".join(wrapped_lines[start:start + lines_per_page])
        fig.text(0.06, 0.91, chunk, fontsize=fontsize, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def add_figure_page(pdf: PdfPages, title: str, fig_specs: list[tuple[str, str]]):
    """Embed 1 or 2 PNG figures on a dedicated landscape/portrait page with captions."""
    n = len(fig_specs)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 11))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    for ax, (img_rel, caption) in zip(axes, fig_specs):
        img_path = ROOT / img_rel
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(caption, fontsize=9, pad=6)
        else:
            ax.text(0.5, 0.5, f"Missing figure: {img_rel}", ha="center", va="center", fontsize=10)
        ax.axis("off")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def _read_csv_rows(rel_path: str) -> list[dict]:
    p = ROOT / rel_path
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _format_table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None) -> str:
    if col_widths is None:
        col_widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "  "
    hdr_line = sep.join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    dash_line = sep.join("-" * w for w in col_widths)
    body_lines = [sep.join(str(r[i]).ljust(w) for i, w in enumerate(col_widths)) for r in rows]
    return "\n".join([hdr_line, dash_line] + body_lines)


def generate_pdf() -> int:
    print("Generating comprehensive team_02_module2_report.pdf ...")
    sqnr_rows = _read_csv_rows("results/mission2/sqnr_measurements.csv")
    word_sqnr = next((r["sqnr_db"] for r in sqnr_rows if r["clip"] == "word_on" and r["fmt"] == "i16"), "76.94")
    sil_sqnr = next((r["sqnr_db"] for r in sqnr_rows if r["clip"] == "silence" and r["fmt"] == "i16"), "70.68")

    # 1. One-Page Client Commitment Memo
    commitment_memo = f"""
TEAM 02 — EDGE EXPERTS | CLIENT COMMITMENT MEMO
Hardware: ESP32-S3-DevKitC-1 N16R8 (240 MHz, 16 MB Flash, 8 MB PSRAM)

1. Audio Format, Sampling Rate, Filter & Scale:
   - Selected: 16,000 Hz, int16, Scale S=1.0, antialias=True.
   - Measured SQNR: {word_sqnr} dB (word) / {sil_sqnr} dB (silence), 0 clipped samples.
   - Pre-hearing scale: S=1.0 uses full [-1, 1] without clipping; int8 S=0.25 clipped 116 samples.

2. Transport Port, Selected Operating Speed & Margin Over Real-Time:
   - Selected operating speed: 1.0x real-time on COM5 (UART) at 921,600 baud, samples_per_packet=320.
   - Theoretical UART ceiling: 92,160 B/s (10 bits/byte) -> 2.823x max with 13 B/640 B framing.
   - Maximum sustainable zero-loss effective speed: 2.737x (feat=False, requested 32.0x) and
     2.753x (feat=True) on COM5; 5.713x (feat=False, requested 6.0x) on COM10. Margin: ~2.74x.

3. Storage Location, Size, Arduino IDE Tools Settings & Remaining Space:
   - Live buffer: PSRAM (8,388,608 B), persisted to LittleFS flash via SAVE.
   - Recording: 144 clips = 2,304,000 samples = 4,608,000 B (CRC 0x4c3c7945).
   - Custom partitions.csv: 9.875 MiB LittleFS (10,354,688 B total).
   - Nominal remaining capacity calculated as filesystem total minus recording size: 5,746,688 B.
   - Arduino IDE Tools: ESP32S3 Dev Module, USB CDC On Boot=Enabled, OPI PSRAM,
     Flash Size=16MB (128Mb), Partition Scheme=Custom.

4. Feature Settings, Compute Time vs Budget & Return Bandwidth:
   - Log-mel: win=480 (30 ms), hop=320 (20 ms), nfft=512, mels=40, 50 frames/s.
   - Radix-2 FFT median compute: 1,711 us/frame (1,084.6x speedup) = 8.6% of 20,000 us hop.
   - Return bandwidth: Log-Mel-40 = 8,400 B/s payload / 9,050 B/s wire;
     MFCC-13 (10 ms hop, 100 fps) = 6,000 B/s payload / 7,300 B/s wire.

5. Loss Accounting:
   - Clean 144-clip: 7,200 pkts, 7,201 frames_ok (incl. EOS), 2,304,000 samples, 7,199 features,
     0 lost, 0 bad_crc, 0 bad_crc_back, 0 seq_gaps.
   - Unexpected loss: fail-closed rejection prevents silent compressed-timeline labelling.

6. Laptop-vs-Board Difference & Classification Impact:
   - Board-vs-NumPy on identical quantised int16 inputs: 0.000286 dB max (satisfies 0.05 dB limit).
   - Original float32 laptop vs board int16: 0.246574 dB max (caused by int16 quantisation;
     0.05 dB implementation tolerance does not apply across different input quantisation).
   - Classifier: 79.17% on both NumPy and Board (0.00% drop). Real-world shifts: int8 S=1.0 -> 54.17%,
     int8 S=0.25 -> 50.00%, 8 kHz no-AA -> 16.67%.

7. Primary Design Risk:
   - Real-world acoustic & microphone domain shift (noise, distance, mic response).
"""

    # Mission 1 Tables
    m1_theo = _format_table(
        ["Format", "Fs (Hz)", "Bytes/Sample", "Payload B/s", "Wire B/s (spp=320)"],
        [
            ["float32", "16000", "4", "64,000", "64,650"],
            ["int16", "16000", "2", "32,000", "32,650"],
            ["int8", "16000", "1", "16,000", "16,650"],
            ["int16", "8000", "2", "16,000", "16,325"],
            ["int8", "8000", "1", "8,000", "8,325"],
        ],
    )
    m1_baud = _format_table(
        ["Port", "Baud", "Payload B/s (measured)", "Wire B/s (measured)", "Max Real-Time (i16)"],
        [
            ["COM5 (UART)", "115,200", "11,358", "11,588", "0.35x"],
            ["COM5 (UART)", "921,600", "89,611", "91,432", "2.80x"],
            ["COM5 (UART)", "2,000,000", "184,745", "188,498", "5.77x"],
            ["COM10 (USB)", "115,200*", "379,419*", "387,126*", "11.86x* (errors)"],
        ],
    )
    m1_pkt = _format_table(
        ["spp", "Payload B", "Overhead %", "Latency (ms)", "COM5 921600 Payload B/s (measured)"],
        [
            ["8", "16", "44.8%", "0.5", "51,014"],
            ["64", "128", "9.2%", "4.0", "83,249"],
            ["320", "640", "2.0%", "20.0", "89,850"],
            ["1024", "2048", "0.6%", "64.0", "91,817"],
        ],
    )
    m1_rt = _format_table(
        ["Config", "Req B/s (wire)", "COM5 115200", "COM5 921600", "COM10 USB"],
        [
            ["16k f32", "64,650", "FAIL", "PASS (1.41x)", "PASS"],
            ["16k i16", "32,650", "FAIL", "PASS (2.80x)", "PASS"],
            ["16k i8", "16,650", "FAIL", "PASS (5.49x)", "PASS"],
            ["8k i16", "16,325", "FAIL", "PASS (5.60x)", "PASS"],
            ["8k i8", "8,325", "PASS (1.39x)", "PASS (11.0x)", "PASS"],
        ],
    )

    # Mission 2 Tables
    m2_sqnr = _format_table(
        ["Clip", "Format", "Scale S", "SQNR (dB)", "Clipped Samples"],
        [[r["clip"], r["fmt"], r["scale"], r["sqnr_db"], r.get("clipped", r.get("clipped_samples", "0"))] for r in sqnr_rows]
        if sqnr_rows else [["word_on", "i16", "1.0", "76.94", "0"], ["silence", "i16", "1.0", "70.68", "0"]],
    )
    m2_mem = _format_table(
        ["Store", "Total Bytes", "Max 16k i16 Duration", "Persists Power-Cycle?"],
        [
            ["SRAM", "~300 KB free", "7.168 s (measured)", "No (volatile)"],
            ["PSRAM (OPI)", "8,388,608 B", "256 s measured (144 clips = 4.608 MB fit)", "No (volatile)"],
            ["LittleFS Flash", "10,354,688 B (9.875 MiB)", "4,608,000 B saved (CRC 0x4c3c7945)", "Yes (verified)"],
        ],
    )

    # Mission 3 Tables
    m3_perf = _format_table(
        ["Implementation", "Stream", "Recv/Exp Feat", "Median us", "p95 us", "Max Err (dB)"],
        [
            ["Starter O(N^2) DFT", "10s COM5 (timeout)", "16 / 499 (483 lost)", "1,855,698", "1,869,197", "0.00168"],
            ["Starter O(N^2) DFT", "10s COM10 (timeout)", "16 / 499 (483 lost)", "1,855,696", "1,881,392", "0.00168"],
            ["Radix-2 FFT", "144s COM5", "7199 / 7199 (0 lost)", "1,711", "1,745", "0.000286"],
        ],
    )
    m3_mfcc = _format_table(
        ["Mode", "Win / Hop", "FPS", "Median us", "Payload B/s", "Wire B/s"],
        [
            ["Log-Mel-40", "480 / 320 (30/20 ms)", "50", "1,711", "8,400", "9,050"],
            ["MFCC-13", "400 / 160 (25/10 ms)", "100", "~1,720", "6,000", "7,300"],
        ],
    )

    # Mission 4 Tables
    m4_spd = _format_table(
        ["Interface", "Requested Speed", "Effective Speed", "Wire Throughput", "Zero-Loss"],
        [
            ["COM5, UART (921600)", "32.0x", "2.737x", "89,360.5 B/s", "PASS (feat=False)"],
            ["COM10, native USB", "6.0x", "5.713x", "186,536.0 B/s", "PASS (feat=False)"],
            ["COM10, native USB", ">=8.0x", "1.54x - 9.65x", "RX overflow", "FAIL (CRC/gaps)"],
        ],
    )
    m4_loss = _format_table(
        ["Stream", "Skip", "Pkts Sent", "Samples Recv", "Feat Recv/Exp", "bad_crc / bcb / gaps"],
        [
            ["Clean 144-clip", "()", "7200/7200", "2,304,000", "7199/7199", "0 / 0 / 0"],
            ["Full 144-clip skip", "(10,11,200)", "7197/7200", "2,303,040", "7196/7199", "0 / 0 / 3"],
            ["6-clip multiclass", "(10,49,150)", "297/300", "95,040", "296/299 (290 valid)", "0 / 0 / 3"],
        ],
    )

    # Mission 5 Tables
    m5_f32 = _format_table(
        ["Comparison", "Max |Delta| (dB)", "Mean |Delta| (dB)", "RMSE (dB)", "p99 |Delta| (dB)"],
        [
            ["A: float32 laptop vs board int16", "0.246574", "0.006220", "0.009785", "0.034024"],
            ["B: numpy int16 vs board int16", "0.000286", "0.000003", "0.000006", "0.000023"],
            ["C: float32 laptop vs numpy int16", "0.246574", "0.006220", "0.009785", "0.034026"],
        ],
    )
    m5_cls = _format_table(
        ["Config", "Train -> Test", "Accuracy", "Balanced Acc", "RMSE vs Ref (dB)"],
        [
            ["16 kHz int16 S=1.0", "NumPy -> Board", "79.17%", "79.17%", "0.000006"],
            ["16 kHz int16 S=1.0", "Float32 -> Board", "79.17%", "79.17%", "0.009785"],
            ["16 kHz int8 S=1.0", "Laptop -> Board", "54.17%", "54.17%", "2.988529"],
            ["16 kHz int8 S=0.25", "Laptop -> Board", "50.00%", "50.00%", "3.366306"],
            ["8 kHz int16 no-AA", "Laptop -> Board", "16.67%", "16.67%", "11.156380"],
        ],
    )

    missions = [
        ("Mission 1: The Pipe (Data Rate & Data Type)", f"""
1. Aim: Measure UART (COM5) and Native USB (COM10) throughput, packet overhead, and real-time feasibility.
2. Background: 16 kHz int16 requires 32,000 B/s; UART sends 10 bits/byte (92,160 B/s raw at 921,600 baud). Each packet adds 13 B framing/CRC.
3. Prediction: 115,200 baud (~11.5 kB/s) cannot carry 16 kHz int16; 921,600 baud (~92.16 kB/s) can carry 16 kHz int16 with >2.5x margin.
4. Method: Ran mission1_pipe.py (3x repeats) across 115200/921600/2000000 baud on COM5 and COM10, plus spp={{8,64,320,1024}} sweep.
5. Observations:
Theoretical Data Rates:
{m1_theo}

Baud-Rate Comparison:
{m1_baud}

Packet-Size Comparison (COM5 921600):
{m1_pkt}

Real-Time Feasibility:
{m1_rt}

6. Analysis: UART is bounded by 92,160 B/s; COM10 native USB is faster but overflows RX buffers at speed=0 without pacing.
   * COM10 115,200 baud row marked with asterisk: all_integrity_ok=False (100 bad_crc, 145 seq_gaps); throughput figures are unreliable and shown for reference only.
   Wire B/s = payload B/s + (1 packet per 20 ms) × 13 B framing = payload + 650 B/s for spp=320 i16 at 16 kHz.
7. What we learnt: COM5 at 921,600 baud with spp=320 balances 2.0% overhead and 20 ms latency reliably.
8. LLM log: Historical pre-experiment LLM prediction was not recorded. Used LLM to write mission1_pipe.py and plot throughput/overhead; all runs executed on hardware.
"""),
        ("Mission 2: Sampling, Quantisation & Persistent Storage", f"""
1. Aim: Select sampling rate, quantisation format/scale, and implement persistent flash storage across power cycles.
2. Background: Downsampling without anti-aliasing folds energy above Nyquist; PSRAM is volatile while LittleFS flash persists.
3. Prediction: 16 kHz int16 S=1.0 avoids aliasing and clipping (>70 dB SQNR); 144 clips (4.608 MB) fit in 8 MB PSRAM and 9.875 MiB LittleFS.
4. Method: Implemented quantise()/pack()/unpack() in mission2_sampling_storage.py, cmdSave() in .ino, custom partitions.csv (10,354,688 B).
5. Observations:
SQNR Measurements:
{m2_sqnr}

Memory & Storage Capacity:
{m2_mem}

Power-Cycle Verification: Saved 4,608,000 B to LittleFS (CRC 0x4c3c7945); after physical USB power-cycle, PSRAM was empty while LittleFS DUMP matched byte-for-byte with CRC 0x4c3c7945.
6. Analysis: Default Arduino partition schemes lacked a ~9.9 MB LittleFS partition; custom partitions.csv solved persistence cleanly.
7. What we learnt: Anti-aliasing must precede downsampling, and LittleFS persistence must be verified across physical power removal.
8. LLM log: Historical pre-experiment LLM prediction was not recorded. Used LLM to write mission2_sampling_storage.py, cmdSave(), and partitions.csv; verified on hardware.
"""),
        ("Mission 3: Real-Time Feature Extraction & FFT Optimization", f"""
1. Aim: Match NumPy log-mel features within 0.05 dB on-board in real time (<20 ms hop budget) and measure speed bottlenecks.
2. Background: Starter O(N^2) DFT took ~1.86 s/frame, dropping 92/99 frames at speed=1.0. Radix-2 FFT reduces complexity to O(N log N).
3. Prediction: Precomputed twiddle/bit-reversal Radix-2 FFT will run in <3 ms/frame (<15% of 20 ms hop) with <0.001 dB error.
4. Method: Replaced dft() in features.h with in-place Radix-2 FFT; ran mission3_features.py and mission3_speed_sweep.py on COM5/COM10.
5. Observations:
Baseline vs Optimised Performance:
{m3_perf}

MFCC-13 & Bandwidth Comparison:
{m3_mfcc}

Streaming Speed Comparison: COM5 effective speed: 2.756x (feat=False) and 2.753x (feat=True). COM10: 3.868x (feat=False) and 3.873x (feat=True).
Baseline Status: Original starter DFT was benchmarked for 2s on COM5 (7/99 frames, 1,855,700 us median).
6. Analysis: COM5 achieves ~2.76x both with and without features, proving UART (92,160 B/s) is the primary bottleneck, not FFT compute.
7. What we learnt: Precomputed Radix-2 FFT makes 50 Hz log-mel extraction effortless on ESP32-S3; UART bandwidth sets the ceiling.
8. LLM log: Historical pre-experiment LLM prediction was not recorded. Used LLM to implement Radix-2 FFT in features.h, mission3_features.py, and mission3_speed_sweep.py.
"""),
        ("Mission 4: Loss Accounting & Dataset Construction", f"""
1. Aim: Account for packet/sample/feature losses, reconstruct compressed timelines, and build a clean 144-clip labelled dataset.
2. Background: Board st.samples only counts received samples, compressing first_sample after gaps; 143 frames cross 1s clip boundaries.
3. Prediction: Deliberate skip=(10,11,200) will cause 3 seq_gaps and 3 lost features; excluding 143 boundary frames yields (144, 49, 40).
4. Method: Ran mission4_loss_dataset.py and mission4_audit.py; unified both entry points via _shared_dataset_workflow() calling build_loss_aware_dataset() and save_dataset_npz().
5. Observations:
Requested vs Effective Streaming Speed (feat=False):
{m4_spd}

Clean vs Deliberate-Loss Streams:
{m4_loss}

Dataset Schema: X float32 (144,49,40), y int64 (144,), clip_ids int64 (144,), classes <U8 (6,), first_sample int64 (144,49), frame_numbers int64 (144,49).
Regression Tests: python -m pytest test_mission4_loss_handling.py -q -> 32 passed, 0 failed.
6. Analysis: Host-side timeline reconstruction from known skip tuples invalidates gap-crossing frames; fail-closed validation rejects any unexpected corruption even when skip_tuple is provided.
7. What we learnt: Aggregate seq_gaps cannot locate unexpected losses; clean datasets require explicit boundary-frame exclusion and universal integrity checks.
8. LLM log: Historical pre-experiment LLM prediction was not recorded. Used LLM to implement mission4_loss_dataset.py, mission4_audit.py, test_mission4_loss_handling.py, and data_card.md.
"""),
        ("Mission 5: Laptop vs Board, Domain Shift & Classification", f"""
1. Aim: Quantify laptop-vs-board feature discrepancy and evaluate nearest-centroid classification across preprocessing configurations.
2. Background: 32-bit float FFT on ESP32 vs 64-bit NumPy may differ slightly; quantisation (int8) and aliasing (8 kHz) cause larger shifts.
3. Prediction: 16 kHz int16 board features will match NumPy (<0.001 dB) with 0% accuracy drop; int8 and unfiltered 8 kHz will degrade accuracy.
4. Method: Ran mission5_domain_shift.py, mission5_board_experiments.py, and mission5_float32_analysis.py (with explicit metadata alignment validation).
5. Observations:
Float32 vs Quantised NumPy vs Board Discrepancy:
{m5_f32}

Classifier Accuracy Comparison:
{m5_cls}

6. Analysis: The board-versus-NumPy implementation discrepancy on identical quantised int16 inputs is 0.000286 dB maximum, satisfying the assignment's 0.05 dB numerical accuracy requirement. The original float32-to-board maximum discrepancy is 0.246574 dB, primarily caused by quantisation rather than firmware implementation error. The 0.05 dB implementation tolerance does not apply to comparisons involving different input quantisation.
7. What we learnt: 16 kHz int16 S=1.0 antialias=True is optimal (79.17% accuracy, 0% drop). Numerical agreement does not imply real-world acoustic robustness.
8. LLM log: Historical pre-experiment LLM prediction was not recorded. Used LLM to implement mission5_domain_shift.py, mission5_preprocessing.py, mission5_board_experiments.py, and mission5_float32_analysis.py.
"""),
    ]

    figure_pages = [
        ("Mission 1 Figures: Throughput & Packet Overhead", [
            ("results/mission1/throughput_vs_baud.png", "Figure 1.1: Throughput vs Baud Rate (COM5 vs COM10)"),
            ("results/mission1/packet_size_overhead.png", "Figure 1.2: Packet Size vs Framing Overhead & Throughput"),
        ]),
        ("Mission 2 Figures: Aliasing Spectra & SQNR", [
            ("results/mission2/aliasing_spectra.png", "Figure 2.1: 48 kHz vs 8 kHz Aliasing Spectra (With & Without Anti-Aliasing)"),
            ("results/mission2/sqnr_comparison.png", "Figure 2.2: SQNR Comparison Across Formats & Scales"),
        ]),
        ("Mission 3 Figures: FFT Optimization & Speed Sweep", [
            ("results/mission3/baseline_vs_optimized.png", "Figure 3.1: Starter DFT vs Radix-2 FFT Compute Time"),
            ("results/mission3/speed_sweep.png", "Figure 3.2: Streaming Speed Sweep (feat=False vs feat=True)"),
        ]),
        ("Mission 3 Figures: Six-Class Log-Mel Spectrograms", [
            ("results/mission3/per_class_logmel.png", "Figure 3.3: Six-Class Log-Mel Spectrograms (on, off, stop, go, _unknown, _silence)"),
        ]),
        ("Mission 4 Figures: Loss Accounting", [
            ("results/mission4/loss_comparison.png", "Figure 4.1: Deliberate Packet Loss vs Lost Feature Frames"),
        ]),
        ("Mission 5 Figures: Float32 & Board Preprocessing Heatmaps", [
            ("results/mission5/float32_board_heatmap.png", "Figure 5.1: Float32-to-Board Discrepancy Heatmaps (A, B, C)"),
            ("results/mission5/board_experiments_heatmaps.png", "Figure 5.2: Board Preprocessing Heatmaps (int8 S=1, int8 S=0.25, 8 kHz no-AA)"),
        ]),
        ("Mission 5 Figures: Classifier Accuracy & Confusion Matrices", [
            ("results/mission5/preprocessing_accuracy.png", "Figure 5.3: Nearest-Centroid Accuracy Across Configurations"),
            ("results/mission5/confusion_matrices.png", "Figure 5.4: Confusion Matrices (NumPy vs Board)"),
        ]),
    ]

    reflections = """
INDIVIDUAL TEAM MEMBER ROLE REFLECTIONS (Team 02 — Edge Experts)

- Member 1 (Builder): [PENDING TEAM CONFIRMATION — do not submit until confirmed] Built/flashed custom partitions.csv, cmdSave() in
  doorbell_board_starter.ino, and Radix-2 FFT in features.h.
- Member 2 (Measurer): [PENDING TEAM CONFIRMATION — do not submit until confirmed] Executed COM5/COM10 throughput, SQNR,
  6-class accuracy, and 144-clip hardware streams.
- Member 3 (Skeptic): [PENDING TEAM CONFIRMATION — do not submit until confirmed] Audited timeline compression, boundary-frame
  exclusion, and real-world acoustic domain shift limitations.
- Member 4 (Scribe): [PENDING TEAM CONFIRMATION — do not submit until confirmed] Compiled mission records, data card, client
  commitment memo, and reproducibility documentation.

REPRODUCIBILITY & LIVE DEMONSTRATION INSTRUCTIONS
1. Install Python dependencies: pip install numpy pyserial matplotlib
2. Flash doorbell_board_starter/doorbell_board_starter.ino with Arduino IDE Tools:
   Board=ESP32S3 Dev Module, USB CDC On Boot=Enabled, PSRAM=OPI PSRAM,
   Flash Size=16MB (128Mb), Partition Scheme=Custom (partitions.csv).
3. Run live demo: python demo_live.py
   (Streams all 144 clips with feat=True & store="psram", executes SAVE, pauses for
   physical USB power-cycle, then verifies CRC 0x4c3c7945 and byte-for-byte DUMP.)
"""

    page_count = 0
    with PdfPages(PDF_PATH) as pdf:
        add_text_page(pdf, "AI255TBD Edge AI — Module 2: The Doorbell's Ears (Team 02)", commitment_memo, fontsize=8.0)
        for title, body in missions:
            add_text_page(pdf, title, body, fontsize=7.8)
        for fig_title, specs in figure_pages:
            add_figure_page(pdf, fig_title, specs)
        data_card = (ROOT / "results/mission4/data_card.md").read_text(encoding="utf-8")
        add_text_page(pdf, "Dataset Data Card (results/mission4/data_card.md)", data_card, fontsize=8.0)
        add_text_page(pdf, "Team Reflections & Reproducibility", reflections, fontsize=8.2)
        page_count = pdf.get_pagecount()
    print(f"Saved {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes, {page_count} pages)")
    return page_count


def clean_pycache():
    for p in ROOT.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    for p in ROOT.rglob("*.pyc"):
        p.unlink(missing_ok=True)
    gitignore = ROOT / ".gitignore"
    entries = {"__pycache__/", "*.pyc", "*.pyo", "*.pyd"}
    existing = set(gitignore.read_text().splitlines()) if gitignore.exists() else set()
    if not entries.issubset(existing):
        merged = sorted(existing | entries)
        gitignore.write_text("\n".join(merged) + "\n", encoding="utf-8")


def build_package_and_zip():
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True)

    files_to_copy = [
        "mission1_pipe.py", "mission2_sampling_storage.py", "mission3_features.py",
        "mission3_speed_sweep.py", "mission4_loss_dataset.py", "mission4_audit.py",
        "test_mission4_loss_handling.py",
        "mission5_domain_shift.py", "mission5_preprocessing.py", "mission5_board_experiments.py",
        "mission5_float32_analysis.py",
        "package_submission.py",
        "demo_live.py", "board_link.py", "team_audio.py", "TEAMS.csv",
        "speech_pool.npz", "README.md", "INSTALL.md", "requirements.txt",
        "team_02_module2_report.pdf",
    ]
    for f in files_to_copy:
        src = ROOT / f
        if src.exists():
            shutil.copy2(src, PKG / f)

    shutil.copytree(ROOT / "doorbell_board_starter", PKG / "doorbell_board_starter")
    shutil.copytree(ROOT / "results", PKG / "results")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(PKG.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(ROOT))
    print(f"Saved {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes)")

    # Strengthened ZIP verification
    tmp_dir = ROOT / "_verify_zip_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        bad = zf.testzip()
        assert bad is None, f"ZIP CRC error in member: {bad}"
        names = zf.namelist()
        assert not any("__pycache__" in n or n.endswith(".pyc") for n in names), \
            "ZIP contains __pycache__ or .pyc files"
        zf.extractall(tmp_dir)

    mandatory = [
        "team_02_module2/mission1_pipe.py",
        "team_02_module2/mission2_sampling_storage.py",
        "team_02_module2/mission3_features.py",
        "team_02_module2/mission3_speed_sweep.py",
        "team_02_module2/mission4_loss_dataset.py",
        "team_02_module2/mission4_audit.py",
        "team_02_module2/test_mission4_loss_handling.py",
        "team_02_module2/mission5_domain_shift.py",
        "team_02_module2/mission5_preprocessing.py",
        "team_02_module2/mission5_board_experiments.py",
        "team_02_module2/mission5_float32_analysis.py",
        "team_02_module2/package_submission.py",
        "team_02_module2/demo_live.py",
        "team_02_module2/board_link.py",
        "team_02_module2/team_audio.py",
        "team_02_module2/speech_pool.npz",
        "team_02_module2/README.md",
        "team_02_module2/INSTALL.md",
        "team_02_module2/requirements.txt",
        "team_02_module2/team_02_module2_report.pdf",
        "team_02_module2/doorbell_board_starter/doorbell_board_starter.ino",
        "team_02_module2/doorbell_board_starter/features.h",
        "team_02_module2/doorbell_board_starter/partitions.csv",
        "team_02_module2/results/mission4/dataset_board.npz",
        "team_02_module2/results/mission4/dataset_numpy.npz",
        "team_02_module2/results/demo_live_report.json",
        "team_02_module2/results/mission5/float32_board_discrepancy.csv",
    ]
    for req in mandatory:
        assert (tmp_dir / req).exists(), f"Missing in ZIP: {req}"

    # Hash comparison for key source files
    for f in ["INSTALL.md", "requirements.txt", "mission4_loss_dataset.py", "team_02_module2_report.pdf"]:
        h_src = hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        h_zip = hashlib.sha256((tmp_dir / "team_02_module2" / f).read_bytes()).hexdigest()
        assert h_src == h_zip, f"Hash mismatch for {f}"

    # Validate both NPZ datasets from extracted ZIP
    import numpy as np
    for npz_rel in ["results/mission4/dataset_board.npz", "results/mission4/dataset_numpy.npz"]:
        with np.load(tmp_dir / "team_02_module2" / npz_rel, allow_pickle=False) as d:
            assert d["X"].shape == (144, 49, 40), f"{npz_rel} X shape: {d['X'].shape}"
            assert d["first_sample"].shape == (144, 49)
            assert d["frame_numbers"].shape == (144, 49)
            assert np.isfinite(d["X"]).all()

    # Run offline pytest from extracted directory
    res = subprocess.run(
        ["python", "-m", "pytest", "test_mission4_loss_handling.py", "-q"],
        cwd=str(tmp_dir / "team_02_module2"),
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"Extracted pytest failed:\n{res.stdout}\n{res.stderr}"

    shutil.rmtree(tmp_dir)
    print("ZIP extraction, hash, NPZ, and offline pytest verification PASSED.")


if __name__ == "__main__":
    if not (ROOT / "results").exists():
        print("Running inside extracted package; skipping full build.")
    else:
        generate_pdf()
        clean_pycache()
        build_package_and_zip()
