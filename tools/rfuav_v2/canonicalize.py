"""canonicalize.py — slice, SNR-estimate, and SNR-adjust raw IQ recordings.

Pipeline
~~~~~~~~
1. Read a raw .iq file (float32 interleaved I/Q)
2. Slice into 0.1 s windows (10M samples @ 100 MHz)
3. Estimate native SNR of each slice via adaptive FCME
4. Write original slices + SNR sidecar to canonical tree
5. On-demand: generate SNR-adjusted slices (AWGN degradation only)

Output layout
~~~~~~~~~~~~~
canonical/
  <taxonomy_id>/
    <recording_stem>/
      slice_0000.iq         # original (float32 interleaved)
      slice_0000_m10dB.iq   # SNR-adjusted (generated on demand)
      slice_0000_snr.json   # {"native_snr_db": 14.5, "fcme_T": 2.5, ...}
      ...

Usage
~~~~~
  python tools/rfuav_v2/canonicalize.py --records assignment.csv  # slice + SNR
  python tools/rfuav_v2/canonicalize.py --records ... --snr-targets -10,0,10  # + SNR adjust
"""

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ChannelSimulation.viewer.signal_proc.snr_dual import FCMEEstimator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 100e6
SLICE_DURATION_S = 0.1
SLICE_SAMPLES = int(FS * SLICE_DURATION_S)  # 10M complex samples
FCME_MAX_SAMPLES = 5_000_000                # 5M samples for SNR estimation
CANONICAL_ROOT = r"E:\Drone_dataset\RFUAV-v2\canonical"
RAW_BASE = r"E:\Drone_dataset\RFUAV"
ASSIGNMENT_CSV = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"


# ---------------------------------------------------------------------------
# Slice a file
# ---------------------------------------------------------------------------

def _count_complex_samples(filepath: str) -> int:
    """Count complex samples in a float32 interleaved IQ file."""
    total_bytes = os.path.getsize(filepath)
    bytes_per_sample = 2 * np.dtype(np.float32).itemsize  # 8 bytes
    return total_bytes // bytes_per_sample


def slice_file(filepath: str, out_dir: str, dry_run: bool = False) -> dict:
    """Slice a raw .iq file into 0.1 s windows.

    Returns
    -------
    stats : dict
        n_slices, total_samples, skipped_bytes
    """
    total = _count_complex_samples(filepath)
    n_slices = total // SLICE_SAMPLES
    if n_slices == 0:
        return {"n_slices": 0, "total_samples": total, "skipped_bytes": total * 8}

    os.makedirs(out_dir, exist_ok=True)
    stats = {"n_slices": n_slices, "total_samples": total, "skipped_bytes": 0}

    if dry_run:
        return stats

    bytes_per_slice = SLICE_SAMPLES * 8  # 2 × float32 × 10M = 80 MB
    with open(filepath, "rb") as src:
        for i in range(n_slices):
            data = src.read(bytes_per_slice)
            if len(data) < bytes_per_slice:
                break
            out_path = os.path.join(out_dir, "slice_%04d.iq" % i)
            # Only write if the slice doesn't already exist
            if not os.path.exists(out_path):
                with open(out_path, "wb") as dst:
                    dst.write(data)

    return stats


# ---------------------------------------------------------------------------
# Estimate SNR of a slice
# ---------------------------------------------------------------------------

def estimate_slice_snr(filepath: str) -> dict:
    """Estimate SNR of a single .iq slice using adaptive FCME.

    Returns
    -------
    dict with keys: native_snr_db, signal_bins, noise_bins, iterations, T_factor
    """
    raw = np.fromfile(filepath, dtype=np.float32, count=FCME_MAX_SAMPLES * 2)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw

    est = FCMEEstimator(fs=FS, nfft=16384)
    snr_db, diag = est.estimate(iq)

    return {
        "native_snr_db": round(float(snr_db), 2),
        "signal_bins": diag["signal_bins"],
        "noise_bins": diag["noise_bins"],
        "iterations": diag["iterations"],
        "T_factor": diag.get("T_factor", 0),
    }


# ---------------------------------------------------------------------------
# SNR adjustment — AWGN degradation only
# ---------------------------------------------------------------------------

def adjust_snr(slice_path: str, target_snr_db: float, native_snr_db: float,
               out_path: str):
    """Add AWGN to degrade a slice to ``target_snr_db``.

    Only degrades: if ``target_snr_db > native_snr_db``, the target is capped
    at ``native_snr_db`` (no denoising).  The output is a float32 interleaved
    .iq file, byte-identical to the original format.

    Parameters
    ----------
    slice_path : str
        Path to the original slice (.iq, float32 interleaved).
    target_snr_db : float
        Desired SNR in dB.
    native_snr_db : float
        Measured native SNR of the slice.
    out_path : str
        Output path for the adjusted slice.
    """
    actual_target = min(target_snr_db, native_snr_db)
    if os.path.exists(out_path):
        return

    raw = np.fromfile(slice_path, dtype=np.float32)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw

    if actual_target >= native_snr_db - 0.01:
        # No degradation needed — just copy the original
        data = np.empty(2 * n, dtype=np.float32)
        data[0::2] = iq.real
        data[1::2] = iq.imag
        data.tofile(out_path)
        return

    # Compute required noise power
    # native_snr = P_sig / P_noise_native
    # target_snr  = P_sig / (P_noise_native + P_noise_added)
    # → P_noise_added = P_sig/target_snr - P_sig/native_snr
    p_sig = float(np.mean(np.abs(iq) ** 2))
    p_noise_native = p_sig / (10 ** (native_snr_db / 10))
    p_noise_target = p_sig / (10 ** (actual_target / 10))
    p_noise_add = max(p_noise_target - p_noise_native, 0)

    if p_noise_add > 0:
        noise_real = np.random.randn(n).astype(np.float32) * np.sqrt(p_noise_add / 2)
        noise_imag = np.random.randn(n).astype(np.float32) * np.sqrt(p_noise_add / 2)
        iq.real += noise_real
        iq.imag += noise_imag

    data = np.empty(2 * n, dtype=np.float32)
    data[0::2] = iq.real
    data[1::2] = iq.imag
    data.tofile(out_path)


# ---------------------------------------------------------------------------
# Process one recording
# ---------------------------------------------------------------------------

def process_recording(row: dict, snr_targets: list, dry_run: bool = False) -> dict:
    """Slice, estimate SNR, and optionally adjust SNR for one recording.

    Parameters
    ----------
    row : dict
        One row from assignment.csv (must have taxonomy_id, source_relpath).
    snr_targets : list[float]
        SNR levels to generate (e.g. [-10, -5, 0, 10, 20]).
    dry_run : bool
        If True, only report what would be done.

    Returns
    -------
    stats : dict
    """
    taxonomy_id = row["taxonomy_id"]
    relpath = row["source_relpath"]
    iq_name = relpath.replace("/", "\\").split("\\")[-1]
    recording_stem = os.path.splitext(iq_name)[0]
    src = os.path.normpath(os.path.join(RAW_BASE, relpath))
    out_dir = os.path.join(CANONICAL_ROOT, taxonomy_id, recording_stem)

    if not os.path.exists(src):
        return {"status": "missing", "recording": recording_stem}

    # ── Step 1: Slice ──
    slice_stats = slice_file(src, out_dir, dry_run=dry_run)
    if slice_stats["n_slices"] == 0:
        return {"status": "empty", "recording": recording_stem, **slice_stats}

    if dry_run:
        return {"status": "dry_run", "recording": recording_stem, **slice_stats}

    # ── Step 2: Estimate SNR on first slice ──
    first_slice = os.path.join(out_dir, "slice_0000.iq")
    snr_info = estimate_slice_snr(first_slice)
    snr_path = os.path.join(out_dir, "slice_0000_snr.json")
    with open(snr_path, "w") as f:
        json.dump(snr_info, f, indent=2)

    native_snr = snr_info["native_snr_db"]

    # ── Step 3: Generate SNR-adjusted slices ──
    generated = []
    for target in snr_targets:
        actual = min(target, native_snr)
        out_name = "slice_0000_%+ddB.iq" % int(actual)
        out_path = os.path.join(out_dir, out_name)
        adjust_snr(first_slice, target, native_snr, out_path)
        generated.append(actual)

    return {
        "status": "ok",
        "recording": recording_stem,
        "n_slices": slice_stats["n_slices"],
        "native_snr_db": native_snr,
        "snr_targets_generated": generated,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_records(csv_path: str, source_root: str = "rawdata") -> list:
    with io.open(csv_path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f)
                if r.get("source_root", "") == source_root
                and r.get("split", "") in ("train", "test")]


def main():
    ap = argparse.ArgumentParser(description="Canonicalize raw IQ recordings")
    ap.add_argument("--records", default=ASSIGNMENT_CSV,
                    help="Path to assignment.csv (default: canonical P0 assignment)")
    ap.add_argument("--snr-targets", default="-10,-5,0,10,20",
                    help="Comma-separated SNR targets in dB (default: -10,-5,0,10,20)")
    ap.add_argument("--workers", type=int, default=2,
                    help="Parallel workers (default: 2)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be done without writing files")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N recordings (0 = all)")
    args = ap.parse_args()

    snr_targets = [float(x) for x in args.snr_targets.split(",")]
    records = load_records(args.records)
    if args.limit:
        records = records[:args.limit]

    print("recordings: %d  |  SNR targets: %s" % (len(records), snr_targets))
    print("output: %s" % CANONICAL_ROOT)

    if args.dry_run:
        # Estimate total slices
        total_slices = 0
        total_gb = 0
        for r in records:
            relpath = r["source_relpath"]
            src = os.path.normpath(os.path.join(RAW_BASE, relpath))
            if not os.path.exists(src):
                continue
            n = _count_complex_samples(src) // SLICE_SAMPLES
            total_slices += n
            total_gb += n * SLICE_SAMPLES * 8 / 1e9
        print("estimated slices: %d  |  raw data: %.1f GB" % (total_slices, total_gb))
        print("per-slice size: %.0f MB (%.0f MB per SNR variant)" % (
            SLICE_SAMPLES * 8 / 1e6, SLICE_SAMPLES * 8 / 1e6 * len(snr_targets)))
        return

    ok = err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_recording, r, snr_targets): i
                for i, r in enumerate(records, 1)}
        for fut in as_completed(futs):
            s = fut.result()
            if s["status"] == "ok":
                ok += 1
                print("[%3d/%3d] OK  %-30s  native SNR=%.1f dB  slices=%d  targets=%s" % (
                    ok + err, len(records), s["recording"][:30],
                    s["native_snr_db"], s["n_slices"],
                    s["snr_targets_generated"]))
            else:
                err += 1
                print("[%3d/%3d] %-6s  %s" % (ok + err, len(records),
                      s["status"], s["recording"][:50]))

    elapsed = time.time() - t0
    print("done: %d ok, %d err  |  %.0f s" % (ok, err, elapsed))


if __name__ == "__main__":
    main()