"""Quarantine DroneRFa format-twin recordings (same capture imported twice:
int32 original + float32 conversion; slice_0000 corr = 1.0000 measured by
recon2_rfa_verify.py). The int32-named recording is kept; the float32-named
duplicate moves to quarantine (never deleted). One kept/quarantined pair had
straddled train/test in manifest v1 (DJIMavic3_float32_RF1 train vs
Mavic3_RF1 test) = byte-level leakage, fixed by this removal."""
import os, json, shutil

ROOT = r"E:\Drone_dataset\RFUAV-v2"
CANON = os.path.join(ROOT, "canonical")
IMAGES = os.path.join(ROOT, "images_canonical")
QDIR = os.path.join(ROOT, "quarantine", "dronerfa_format_twins")

TWINS = [  # (tax, recording_to_quarantine, kept_duplicate)
    ("uav.dji.avata2", "AVATA_T1110_S1000_RF0_float32_rf0", "AVATA_T1110_S1000_RF0_rf0"),
    ("uav.dji.avata2", "AVATA_T1110_S1000_RF1_float32_rf1", "AVATA_T1110_S1000_RF1_rf1"),
    ("uav.dji.mavic3", "DJIMavic3_float32_RF0_rf0", "Mavic3_RF0_rf0"),
    ("uav.dji.mavic3", "DJIMavic3_float32_RF1_rf1", "Mavic3_RF1_rf1"),
]

moved = []
for tax, rec, keep in TWINS:
    src = os.path.join(CANON, tax, rec)
    if not os.path.isdir(src):
        print("SKIP already moved: %s/%s" % (tax, rec))
        continue
    if not os.path.isdir(os.path.join(CANON, tax, keep)):
        raise SystemExit("ABORT: kept twin missing: %s/%s" % (tax, keep))
    dst = os.path.join(QDIR, "canonical", tax, rec)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    isrc = os.path.join(IMAGES, tax, rec)
    if os.path.isdir(isrc):
        idst = os.path.join(QDIR, "images", tax, rec)
        os.makedirs(os.path.dirname(idst), exist_ok=True)
        shutil.move(isrc, idst)
    moved.append({"taxonomy_id": tax, "recording": rec,
                  "kept_duplicate": keep, "slice0_corr": 1.0000})
    print("quarantined %s/%s (dup of %s)" % (tax, rec, keep))

os.makedirs(QDIR, exist_ok=True)
json.dump({
    "reason": "format twins: same DroneRFa capture imported twice (int32 "
              "original + float32 conversion), slice_0000 corr=1.0000; kept "
              "the int32-named recording; one pair straddled train/test in "
              "manifest v1 (byte-level leakage, now removed)",
    "measured_by": "tools/recon2_rfa_verify.py",
    "date": "2026-08-16",
    "moved": moved,
}, open(os.path.join(QDIR, "README.json"), "w", encoding="utf-8"), indent=2)
print("quarantine done: %d moved" % len(moved))
