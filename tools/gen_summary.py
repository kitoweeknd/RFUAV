"""Regenerate canonical_summary.md from the current master manifest."""
import csv, time
from collections import defaultdict

MANIFEST = r"E:\Drone_dataset\RFUAV-v2\manifests\canonical_manifest_v1.csv"
OUT = r"E:\Drone_dataset\RFUAV-v2\manifests\canonical_summary.md"
NON_TRANSMITTER = {"bg.kong", "ext.dronerfb.background", "unk.weizhi", "bg.dronerf"}

rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))

by_class = defaultdict(list)
by_src = defaultdict(lambda: defaultdict(int))
for r in rows:
    by_class[r["taxonomy_id"]].append(r)
    by_src[r["source_dataset"]][r["split"]] += 1

n_slices = sum(int(r["n_slices"] or 0) for r in rows)
n_images = sum(int(r["n_images"] or 0) for r in rows)
cls_split = {c: set(x["split"] for x in v) for c, v in by_class.items()}
both = sorted(c for c, s in cls_split.items() if "train" in s and "test" in s)
train_only = sorted(c for c, s in cls_split.items() if "train" in s and "test" not in s)
n_emit = sum(1 for c in by_class if c not in NON_TRANSMITTER)

L = []
L.append("# RFUAV-v2 Canonical Dataset Summary (updated %s)" % time.strftime("%Y-%m-%d"))
L.append("")
L.append("Unified format: `canonical/<taxonomy_id>/<recording>/slice_%04d.iq` (float32 interleaved IQ)")
L.append("+ `metadata.json` sidecar (source, split, FCME native SNR, sample rate, provenance).")
L.append("Rendered tree: `images_canonical/` — 1 spectrogram per slice, 1920x1440 JPEG q95, STFT-1024.")
L.append("")
L.append("| metric | value |")
L.append("|---|---|")
L.append("| taxonomy classes | %d (%d drone/RC emitters + 2 background + 1 unknown-isolated) |"
         % (len(by_class), n_emit))
models = set()
for c in by_class:
    if c.startswith("ext.dronerfb.") and c[-1].isdigit():
        models.add("ext.dronerfb.model_" + c.split(".")[-1][0])
    else:
        models.add(c)
n_model_emit = sum(1 for m in models if m not in NON_TRANSMITTER)
L.append("| model-level classes | %d total / %d emitters (DroneRFb 18 individuals collapse to 6 models) |"
         % (len(models), n_model_emit))
L.append("| classes with train+test (evaluable) | %d |" % len(both))
L.append("| classes train-only | %d |" % len(train_only))
L.append("| recordings | %d |" % len(rows))
L.append("| slices / images | %d / %d |" % (n_slices, n_images))
L.append("")
L.append("| source | train | test | excluded | split basis |")
L.append("|---|---|---|---|---|")
basis = {"DroneRF": "flight-mode session-grouped (L/H bands same side)",
         "DroneRFa": "session-grouped, RF-chain token stripped (v2 fix)",
         "DroneRFb-DIR": "cross-environment (IN=train, OUT=test)",
         "RFUAV": "P0 recording-grouped assignment"}
for s in sorted(by_src):
    d = by_src[s]
    L.append("| %s | %d | %d | %d | %s |" % (
        s, d.get("train", 0), d.get("test", 0), d.get("excluded", 0), basis.get(s, "")))
L.append("")
L.append("Decisions of record:")
L.append("- PublicDataset slice-only sessions (30) NOT materialized (E-013: derived slices,")
L.append("  contamination risk). 30 sidecar-only orphan dirs in `quarantine/canonical_orphans/`.")
L.append("- DroneRFb test/ recovered via test_labels.txt -> +6 third-individual classes (A3-G3).")
L.append("- FCME adaptive NaN bug fixed (noise-set collapse on full-band signals); 13 recordings")
L.append("  re-estimated.")
L.append("- 2026-08-16 DroneRFa TOP-UP: all 54 recordings extended 2 -> 10 slices from verified")
L.append("  source readers (slice_0000/0001 corr=1.0000; int32 sources use importer scale 2^-31,")
L.append("  float32 sources scale 1.0). +432 slices, all rendered.")
L.append("- 2026-08-16 FORMAT TWINS quarantined: 4 DroneRFa recordings were duplicate imports of")
L.append("  the same capture (int32 original + float32 conversion, corr=1.0000). One pair")
L.append("  straddled train/test in manifest v1 = byte-level leakage. Float32-named copies moved")
L.append("  to `quarantine/dronerfa_format_twins/`; int32-named recordings kept.")
L.append("- 2026-08-16 SPLIT FIX (manifest v2): DroneRFa now groups by SESSION (RF0/RF1 antenna")
L.append("  token stripped from original_file). v1 had grouped by raw original_file, putting the")
L.append("  two antennas of one capture on opposite sides for 4 captures (session leakage) and")
L.append("  propping up evaluable classes at 61. Honest count after fix: %d evaluable" % len(both))
L.append("  (single-session DroneRFa-only classes are train_only; recoverable later via an")
L.append("  explicit cross-source eval tier, not by splitting sessions).")
L.append("")
L.append("## Augmented data asset (train-side only, valid untouched)")
L.append("")
L.append("`augmented/iq/` holds one channel-augmented copy of every train-side")
L.append("slice of the evaluable classes (float32 IQ kept per user directive,")
L.append("~151 GB) + `augmented/images/` spectrograms + per-slice params in")
L.append("`aug_params.jsonl` (deterministic sha256 seeds -> bit-exact")
L.append("regenerable). Full accounting: `manifests/augmented_manifest_v1.csv`.")
L.append("Chain: fading/doppler/multipath/CFO/phase-noise/IQ-imbalance/AWGN.")
L.append("")
L.append("## Per-class table")
L.append("")
L.append("| class | sources | recs | slices | train | test | SNR (dB) |")
L.append("|---|---|---|---|---|---|---|")
for c in sorted(by_class):
    v = by_class[c]
    srcs = "+".join(sorted(set(x["source_dataset"] for x in v)))
    sl = sum(int(x["n_slices"] or 0) for x in v)
    tr = sum(1 for x in v if x["split"] == "train")
    te = sum(1 for x in v if x["split"] == "test")
    snrs = [float(x["native_snr_db"]) for x in v if x["native_snr_db"]]
    srng = "%.1f..%.1f" % (min(snrs), max(snrs)) if snrs else "-"
    L.append("| %s | %s | %d | %d | %d | %d | %s |" % (c, srcs, len(v), sl, tr, te, srng))
L.append("")
L.append("Train/valid image tree: `train_dataset_v1/` (%d evaluable classes, hardlinked from" % len(both))
L.append("`images_canonical/`, recording-grouped, 0 straddle) — see `train_split_v1_report.json`.")
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("summary regenerated: %s (%d classes, %d rows)" % (OUT, len(by_class), len(rows)))
