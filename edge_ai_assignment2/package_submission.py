"""
package_submission.py -- Generate final 8-heading PDF report, clean pycache,
                         build team_02_module2/ and team_02_module2.zip, and verify.
Team 02 | AI255TBD Edge AI Module 2
"""

import shutil
import textwrap
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(".")
PKG = ROOT / "team_02_module2"
ZIP_PATH = ROOT / "team_02_module2.zip"
PDF_PATH = ROOT / "team_02_module2_report.pdf"


def add_text_page(pdf: PdfPages, title: str, body: str, fontsize: int = 8.5):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.94, title, fontsize=13, fontweight="bold", va="top")
    wrapped_lines = []
    for line in body.splitlines():
        if not line.strip():
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=98, replace_whitespace=False))
    # Paginate if needed
    lines_per_page = 62
    for start in range(0, max(1, len(wrapped_lines)), lines_per_page):
        if start > 0:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.07, 0.94, f"{title} (cont.)", fontsize=11, fontweight="bold", va="top")
        chunk = "\n".join(wrapped_lines[start:start + lines_per_page])
        fig.text(0.07, 0.90, chunk, fontsize=fontsize, family="monospace", va="top")
    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf():
    print("Generating team_02_module2_report.pdf ...")
    import csv
    with open(ROOT / "results/mission2/sqnr_measurements.csv") as f:
        sqnr_rows = list(csv.DictReader(f))
    word_sqnr = next(r["sqnr_db"] for r in sqnr_rows if r["clip"] == "word_on" and r["fmt"] == "i16")
    sil_sqnr = next(r["sqnr_db"] for r in sqnr_rows if r["clip"] == "silence" and r["fmt"] == "i16")

    commitment_memo = f"""
TEAM 02 — EDGE EXPERTS | CLIENT COMMITMENT MEMO
Hardware: ESP32-S3-DevKitC-1 N16R8 (240 MHz, 16 MB Flash, 8 MB PSRAM)

1. Audio Format, Sampling Rate, Filter & Scale:
   - Selected: 16,000 Hz, int16, Scale S=1.0, antialias=True.
   - Measured SQNR: {word_sqnr} dB (word) / {sil_sqnr} dB (silence), 0 clipped samples.
   - Pre-hearing scale: S=1.0 uses full [-1, 1] without clipping; int8 S=0.25 clipped 116 samples.

2. Transport Port, Speed & Margin Over Real-Time:
   - Selected: COM5 (UART) at 921,600 baud, samples_per_packet=320.
   - Theoretical UART ceiling: 92,160 B/s (10 bits/byte) -> 2.823x max with 13 B/640 B framing.
   - Measured zero-loss effective speed: 2.756x (feat=False), 2.753x (feat=True) on COM5;
     5.713x (feat=False) and 3.873x (feat=True) on COM10. Margin: ~2.75x real-time, 0 errors.

3. Storage Location, Size, Arduino IDE Tools Settings & Remaining Space:
   - Live buffer: PSRAM (8,388,608 B), persisted to LittleFS flash via SAVE.
   - Recording: 144 clips = 2,304,000 samples = 4,608,000 B (CRC 0x4c3c7945).
   - Custom partitions.csv: 9.875 MiB LittleFS (10,354,688 B total); 5,726,208 B free after SAVE.
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
   - Identical int16: max |Delta| = 0.000286 dB; 79.17% on both NumPy and Board (0.00% drop).
   - Real-world shifts: int8 S=1.0 -> 54.17%, int8 S=0.25 -> 50.00%, 8 kHz no-AA -> 16.67%.

7. Primary Design Risk:
   - Real-world acoustic & microphone domain shift (noise, distance, mic response).
"""

    missions = [
        ("Mission 1: The Pipe (Data Rate & Data Type)", """
1. Aim: Measure UART (COM5) and Native USB (COM10) throughput, packet overhead, and real-time feasibility for 16 kHz / 8 kHz audio.
2. Background: 16 kHz int16 requires 32,000 B/s; UART sends 10 bits/byte (92,160 B/s raw at 921,600 baud). Each packet adds 13 B framing/CRC.
3. Prediction: 115,200 baud (~11.5 kB/s) cannot carry 16 kHz int16; 921,600 baud (~92.16 kB/s) can carry 16 kHz int16 with >2.5x margin.
4. Method: Ran mission1_pipe.py (3x repeats) across 115200/921600/2000000 baud on COM5 and COM10, plus spp={8,64,320,1024} sweep.
5. Observations: COM5 921600 achieved ~88 kB/s (~2.75x real-time) with 0 errors; spp=320 balanced overhead (2.0%) and 20 ms latency.
6. Analysis: UART is bounded by 92,160 B/s; COM10 native USB is faster but overflows RX buffers at speed=0 without pacing.
7. What we learnt: COM5 at 921,600 baud with spp=320 is reliable and deterministic for real-time 16 kHz int16 streaming.
8. LLM log: Used LLM to write mission1_pipe.py and plot throughput/overhead; all runs executed on hardware.
"""),
        ("Mission 2: Sampling, Quantisation & Persistent Storage", """
1. Aim: Select sampling rate, quantisation format/scale, and implement persistent flash storage across power cycles.
2. Background: Downsampling without anti-aliasing folds energy above Nyquist; PSRAM is volatile while LittleFS flash persists.
3. Prediction: 16 kHz int16 S=1.0 avoids aliasing and clipping (>70 dB SQNR); 144 clips (4.608 MB) fit in 8 MB PSRAM and 9.875 MiB LittleFS.
4. Method: Implemented quantise()/pack()/unpack() in mission2_sampling_storage.py, cmdSave() in .ino, custom partitions.csv (10,354,688 B).
5. Observations: 16 kHz int16 S=1.0 gave 76.94 dB SQNR (0 clips); 4,608,000 B saved to LittleFS (CRC 0x4c3c7945), verified after physical power cycle.
6. Analysis: Default Arduino partition schemes lacked a ~9.9 MB LittleFS partition; custom partitions.csv solved persistence cleanly.
7. What we learnt: Anti-aliasing must precede downsampling, and LittleFS persistence must be verified across physical power removal.
8. LLM log: Used LLM to write mission2_sampling_storage.py, cmdSave(), and partitions.csv; verified on hardware.
"""),
        ("Mission 3: Real-Time Feature Extraction & FFT Optimization", """
1. Aim: Match NumPy log-mel features within 0.05 dB on-board in real time (<20 ms hop budget) and measure speed bottlenecks.
2. Background: Starter O(N^2) DFT took ~1.86 s/frame, dropping 92/99 frames at speed=1.0. Radix-2 FFT reduces complexity to O(N log N).
3. Prediction: Precomputed twiddle/bit-reversal Radix-2 FFT will run in <3 ms/frame (<15% of 20 ms hop) with <0.001 dB error.
4. Method: Replaced dft() in features.h with in-place Radix-2 FFT; ran mission3_features.py and mission3_speed_sweep.py on COM5/COM10.
5. Observations: Median compute dropped 1,084.6x (1,855,700 -> 1,711 us/frame, 8.6% of hop). 144-clip stream: 7,199/7,199 frames, max err 0.000286 dB.
   COM5 effective speed: 2.756x (feat=False) and 2.753x (feat=True). COM10: 3.868x (feat=False) and 3.873x (feat=True).
6. Analysis: COM5 achieves ~2.76x both with and without features, proving UART (92,160 B/s) is the primary bottleneck, not FFT compute.
7. What we learnt: Precomputed Radix-2 FFT makes 50 Hz log-mel extraction effortless on ESP32-S3; UART bandwidth sets the ceiling.
8. LLM log: Used LLM to implement Radix-2 FFT in features.h, mission3_features.py, and mission3_speed_sweep.py.
"""),
        ("Mission 4: Loss Accounting & Dataset Construction", """
1. Aim: Account for packet/sample/feature losses, reconstruct compressed timelines, and build a clean 144-clip labelled dataset.
2. Background: Board st.samples only counts received samples, compressing first_sample after gaps; 143 frames cross 1s clip boundaries.
3. Prediction: Deliberate skip=(10,11,200) will cause 3 seq_gaps and 3 lost features; excluding 143 boundary frames yields (144, 49, 40).
4. Method: Ran mission4_loss_dataset.py and mission4_audit.py (COM5/COM10 speed sweeps, full 144-clip skip=(10,11,200), clean dataset build).
5. Observations: Clean 144-clip: 7,200 pkts, 7,201 frames_ok, 2,304,000 samples, 7,199 features, 0 errors. Full skip=(10,11,200): 7,197 pkts,
   7,198 frames_ok, 2,303,040 samples, 7,196 features, 3 seq_gaps, 3 lost. Saved first_sample & frame_numbers in both NPZs.
6. Analysis: Host-side timeline reconstruction from known skip tuples is essential to invalidate gap-crossing frames near class transitions.
7. What we learnt: Aggregate seq_gaps cannot locate unexpected losses; clean datasets require explicit boundary-frame exclusion.
8. LLM log: Used LLM to implement mission4_loss_dataset.py, mission4_audit.py, and data_card.md.
"""),
        ("Mission 5: Laptop vs Board, Domain Shift & Classification", """
1. Aim: Quantify laptop-vs-board feature discrepancy and evaluate nearest-centroid classification across preprocessing configurations.
2. Background: 32-bit float FFT on ESP32 vs 64-bit NumPy may differ slightly; quantisation (int8) and aliasing (8 kHz) cause larger shifts.
3. Prediction: 16 kHz int16 board features will match NumPy (<0.001 dB) with 0% accuracy drop; int8 and unfiltered 8 kHz will degrade accuracy.
4. Method: Ran mission5_domain_shift.py and mission5_board_experiments.py (streaming all 144 clips on COM5 for int8 S=1, int8 S=0.25, 8 kHz no-AA).
5. Observations: 16 kHz int16: 0.000286 dB max diff, 79.17% on both NumPy and Board (0% drop). On actual board streams: int8 S=1.0 dropped
   Laptop->Board to 54.17%; int8 S=0.25 dropped to 50.00%; 8 kHz antialias=False collapsed to 16.67% (11.16 dB RMSE vs filtered 8 kHz).
6. Analysis: Numerical implementation discrepancy is negligible; quantisation noise, clipping, and aliasing are the true domain-shift drivers.
7. What we learnt: 16 kHz int16 S=1.0 antialias=True is optimal. Numerical agreement does not imply real-world acoustic robustness.
8. LLM log: Used LLM to implement mission5_domain_shift.py, mission5_preprocessing.py, and mission5_board_experiments.py.
"""),
    ]

    reflections = """
INDIVIDUAL TEAM MEMBER ROLE REFLECTIONS (Team 02 — Edge Experts)

- Member 1 (Builder): [TODO: confirm with team] Built/flashed custom partitions.csv, cmdSave() in
  doorbell_board_starter.ino, and Radix-2 FFT in features.h.
- Member 2 (Measurer): [TODO: confirm with team] Executed COM5/COM10 throughput, SQNR,
  6-class accuracy, and 144-clip hardware streams.
- Member 3 (Skeptic): [TODO: confirm with team] Audited timeline compression, boundary-frame
  exclusion, and real-world acoustic domain shift limitations.
- Member 4 (Scribe): [TODO: confirm with team] Compiled mission records, data card, client
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

    with PdfPages(PDF_PATH) as pdf:
        add_text_page(pdf, "AI255TBD Edge AI — Module 2: The Doorbell's Ears (Team 02)", commitment_memo)
        for title, body in missions:
            add_text_page(pdf, title, body)
        data_card = (ROOT / "results/mission4/data_card.md").read_text(encoding="utf-8")
        add_text_page(pdf, "Dataset Data Card (results/mission4/data_card.md)", data_card)
        add_text_page(pdf, "Team Reflections & Reproducibility", reflections)
    print(f"Saved {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")


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
        "demo_live.py", "board_link.py", "team_audio.py", "TEAMS.csv",
        "speech_pool.npz", "README.md", "INSTALL.md", "requirements.txt",
        "team_02_module2_report.pdf",
    ]
    for f in files_to_copy:
        shutil.copy2(ROOT / f, PKG / f)

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
    import hashlib
    import subprocess
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
    generate_pdf()
    clean_pycache()
    build_package_and_zip()
