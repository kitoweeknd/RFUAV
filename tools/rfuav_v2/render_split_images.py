"""render_split_images.py -- render just the assigned recordings into a clean image tree.

One-shot utility for P0 section5: reads assignment.csv, filters to rawdata source_root +
train/test splits, and calls RawDataProcessor.generate_images for each .iq file.

Output layout: <out_root>/<taxonomy_id>/<iq_basename.iq>/<taxonomy_id> (frame).jpg
This is the SAME nested layout split_images_grouped expects, and the .iq directory
name equals the manifest's source_relpath tail (with extension), so the match is
exact by construction -- no name-bridge hack.

Usage::
  python tools\\rfuav_v2\\render_split_images.py          # 4 workers, full batch
  python tools\\rfuav_v2\\render_split_images.py --dry    # print first 5 files
  python tools\\rfuav_v2\\render_split_images.py 8        # 8 workers
"""

import csv
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
MANIFEST_DIR = r"E:\Drone_dataset\RFUAV-v2\manifests"
ASSIGNMENT   = os.path.join(MANIFEST_DIR, "assignment.csv")
RAW_BASE     = r"E:\Drone_dataset\RFUAV"
OUT_ROOT     = r"E:\Drone_dataset\RFUAV-v2\images_grouped"

# augment sys.path so graphic.RawDataProcessor can be imported
_PROJ = r"D:\ML\RFUAV"
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from graphic.RawDataProcessor import generate_images


# ---------------------------------------------------------------------------
def load_assignment():
    rows = []
    with io.open(ASSIGNMENT, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source_root", "") == "rawdata" and r.get("split", "") in ("train", "test"):
                rows.append(r)
    return rows


def _tail_basename(relpath):
    """Return the last path component of a Windows/POSIX source_relpath."""
    return relpath.replace("/", "\\").split("\\")[-1]


def render_one(row, idx, total):
    taxonomy_id = row["taxonomy_id"]
    relpath     = row["source_relpath"]
    iq_name     = _tail_basename(relpath)          # e.g. "DAUTEL_EVO Nano-70db....iq"
    src         = os.path.normpath(os.path.join(RAW_BASE, relpath))
    size_mb     = int(row.get("size_bytes", "0") or "0") / 1e6

    # skip if already rendered (output dir exists and has .jpg files)
    out_dir = os.path.join(OUT_ROOT, taxonomy_id, iq_name)
    if os.path.isdir(out_dir) and any(f.lower().endswith(".jpg") for f in os.listdir(out_dir)):
        njpg = sum(1 for f in os.listdir(out_dir) if f.lower().endswith(".jpg"))
        print("[%3d/%3d] SKIP %6.0f MB  %s  (%d frames exist)" % (idx, total, size_mb, iq_name[:50], njpg))
        return (idx, True, None)

    t0 = time.time()
    try:
        generate_images(
            datapack=src,
            file=taxonomy_id,
            pack=iq_name,
            fs=100e6,
            stft_point=2048,
            duration_time=0.1,
            step=1,
            location=OUT_ROOT,
        )
        elapsed = time.time() - t0
        print("[%3d/%3d] OK  %6.0f MB  %5.1fs  %s" % (idx, total, size_mb, elapsed, iq_name[:50]))
        return (idx, True, None)
    except Exception as e:
        elapsed = time.time() - t0
        print("[%3d/%3d] ERR %6.0f MB  %5.1fs  %s  -- %s" % (idx, total, size_mb, elapsed, iq_name[:50], e))
        return (idx, False, str(e))


# ---------------------------------------------------------------------------
def main():
    rows = load_assignment()
    print("rawdata train/test recordings: %d" % len(rows))
    total_sz = sum(int(r.get("size_bytes", "0") or "0") for r in rows)
    print("total size: %.1f GB" % (total_sz / 1e9))

    # --- dry mode: just print the first 5 ---
    if "--dry" in sys.argv:
        print("--- dry-run: first 5 ---")
        for i, r in enumerate(rows[:5], 1):
            print("  [%d] %-30s %s" % (i, r["taxonomy_id"][:30], _tail_basename(r["source_relpath"])[:70]))
        return

    workers = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4
    print("workers: %d" % workers)

    ok = 0
    err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(render_one, r, i, len(rows)): i for i, r in enumerate(rows, 1)}
        for fut in as_completed(futs):
            idx, status, msg = fut.result()
            if status:
                ok += 1
            else:
                err += 1
    elapsed = time.time() - t0
    print("---")
    print("done: %d ok, %d err in %.0f s (%.0f min)" % (ok, err, elapsed, elapsed / 60))


if __name__ == "__main__":
    main()
