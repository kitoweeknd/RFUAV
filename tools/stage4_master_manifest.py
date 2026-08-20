"""Stage 4 (v2): build the unified master manifest over the canonical tree.

One row per recording with source, SNR, slice/image counts, and a
leakage-safe suggested split:
  RFUAV      -> split copied from the P0 grouped assignment
  DroneRFb   -> IN sessions train / OUT session test (cross-environment,
                session-grouped); background -> train
  DroneRFa   -> SESSION-grouped within class: the session key strips the
                RF-chain token (_RF0/_RF1) from original_file, so both
                antennas of one capture always land on the same side.
                (v1 grouped by raw original_file, which silently put RF0 in
                train and RF1 in test for several captures = session leakage.)
                Single-session classes are train_only.
"""
import os, re, json, csv, time
from collections import defaultdict

RAW = r"E:\Drone_dataset\RFUAV-v2\rawdata"
IMAGES = r"E:\Drone_dataset\RFUAV-v2\images\spectrogram"
OUT = r"E:\Drone_dataset\RFUAV-v2\manifests\canonical_manifest_v1.csv"
SLICE_RE = re.compile(r"^slice_\d{4}\.iq$")

def log(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

def rfa_session_key(meta, rec):
    """Session key for DroneRFa: original_file with the RF-chain token removed."""
    of = meta.get("original_file") or re.sub(r"_rf\d+$", "", rec)
    return re.sub(r"_RF\d(?=_|$)", "", of, flags=re.IGNORECASE)

import glob as _glob
recs = []
for mpath in _glob.glob(os.path.join(RAW, "*", "*", "*", "metadata.json")):
    rdir = os.path.dirname(mpath)
    rec = os.path.basename(rdir)
    tax = os.path.basename(os.path.dirname(rdir))
    with open(mpath) as f:
        meta = json.load(f)
    n_slices = sum(1 for f in os.listdir(rdir) if SLICE_RE.match(f))
    img_dir = os.path.join(IMAGES, tax, rec)
    n_images = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")]) \
        if os.path.isdir(img_dir) else 0
    recs.append({"tax": tax, "rec": rec, "meta": meta,
                 "n_slices": n_slices, "n_images": n_images})
recs.sort(key=lambda r: (r["tax"], r["rec"]))

log("Stage 4 v2 start: %d recordings" % len(recs))

# DroneRFa grouping: class -> session keys (RF token stripped)
rfa_groups = defaultdict(set)
for r in recs:
    if r["meta"].get("source_dataset") == "DroneRFa":
        rfa_groups[r["tax"]].add(rfa_session_key(r["meta"], r["rec"]))
rfa_test_group = {}
for tax, groups in rfa_groups.items():
    gs = sorted(groups)
    rfa_test_group[tax] = gs[-1] if len(gs) >= 2 else None

# DroneRF grouping: session = flight-mode BUI; L/H bands of one session are
# simultaneous captures and must stay on the same side.
drf_groups = defaultdict(set)
for r in recs:
    if r["meta"].get("source_dataset") == "DroneRF":
        drf_groups[r["tax"]].add(r["meta"].get("session_id", r["rec"]))
drf_test_group = {}
for tax, groups in drf_groups.items():
    gs = sorted(groups)
    drf_test_group[tax] = gs[-1] if len(gs) >= 2 else None

rows = []
for r in recs:
    meta = r["meta"]
    src = meta.get("source_dataset", "RFUAV")
    if src == "RFUAV":
        split = meta.get("split", "") or "unassigned"
        basis = "p0_grouped_assignment"
    elif src == "DroneRFb-DIR":
        env = meta.get("environment", "NA")
        split = "test" if env == "OUT" else "train"
        basis = "cross_environment_session"
    elif src == "DroneRFa":
        sk = rfa_session_key(meta, r["rec"])
        tg = rfa_test_group.get(r["tax"])
        if tg is None:
            split = "train"
            basis = "session_grouped_train_only"
        else:
            split = "test" if sk == tg else "train"
            basis = "session_grouped"
    elif src == "DroneRF":
        sk = meta.get("session_id", r["rec"])
        tg = drf_test_group.get(r["tax"])
        if tg is None:
            split = "train"
            basis = "session_grouped_train_only"
        else:
            split = "test" if sk == tg else "train"
            basis = "session_grouped"
    else:
        split = "unassigned"
        basis = "unknown_source"

    # model-level identity: DroneRFb individuals collapse to their model
    # letter (a1/a2/a3 -> model_a); every other class is its own model.
    if r["tax"].startswith("ext.dronerfb.") and r["tax"][-1].isdigit():
        model_group = "ext.dronerfb.model_" + r["tax"].split(".")[-1][0]
    else:
        model_group = r["tax"]

    rows.append({
        "taxonomy_id": r["tax"],
        "model_group": model_group,
        "recording": r["rec"],
        "source_dataset": src,
        "split": split,
        "split_basis": basis,
        "native_snr_db": meta.get("native_snr_db", ""),
        "sample_rate_hz": meta.get("sample_rate_hz", ""),
        "n_slices": r["n_slices"],
        "n_slices_available": meta.get("n_slices_available", ""),
        "n_images": r["n_images"],
        "environment": meta.get("environment", ""),
        "scene": meta.get("scene", ""),
        "capture_condition": meta.get("capture_condition", ""),
        "session_id": meta.get("session_id", ""),
        "eval_tier": meta.get("eval_tier", ""),
        "source_ref": meta.get("source_relpath") or meta.get("original_file", ""),
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
log("manifest written: %s (%d rows)" % (OUT, len(rows)))

# Summaries
by_src = defaultdict(lambda: defaultdict(int))
cls_split = defaultdict(set)
for row in rows:
    by_src[row["source_dataset"]][row["split"]] += 1
    cls_split[row["taxonomy_id"]].add(row["split"])
log("split counts by source:")
for s in sorted(by_src):
    log("  %-14s %s" % (s, dict(by_src[s])))
both = sum(1 for c, sp in cls_split.items() if "train" in sp and "test" in sp)
train_only = sum(1 for c, sp in cls_split.items() if "train" in sp and "test" not in sp)
log("classes total=%d  with train+test=%d  train_only=%d" % (
    len(cls_split), both, train_only))
models = set(row["model_group"] for row in rows)
log("model-level groups=%d (individual classes=%d)" % (len(models), len(cls_split)))
imgs = sum(r["n_images"] for r in rows)
slices = sum(r["n_slices"] for r in rows)
log("totals: slices=%d images=%d" % (slices, imgs))
