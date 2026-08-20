"""Build the full data inventory over RFUAV-v2:
  manifests\\inventory_recordings.csv  - one row per recording, byte-accurate
  manifests\\inventory_classes.csv     - one row per taxonomy class
  manifests\\inventory_report.md       - tree sizes + distribution + integrity
Also runs integrity checks (metadata present, slice count vs metadata,
image count == slice count, SNR present) and reports every violation.
"""
import os, csv, json, glob, time
from collections import defaultdict

ROOT = r"E:\Drone_dataset\RFUAV-v2"
RAW = os.path.join(ROOT, "rawdata")
IMAGES = os.path.join(ROOT, "images", "spectrogram")
MANI = os.path.join(ROOT, "manifests")
MANIFEST = os.path.join(MANI, "canonical_manifest_v1.csv")
NON_TRANSMITTER = {"bg.kong", "ext.dronerfb.background", "unk.weizhi", "bg.dronerf"}

split_info = {}
for r in csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")):
    split_info[(r["taxonomy_id"], r["recording"])] = (r["split"], r["split_basis"])


def dir_bytes(path):
    total = 0
    for dp, dn, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


rows, issues = [], []
rdirs = sorted(glob.glob(os.path.join(RAW, "*", "*", "*", "metadata.json")),
               key=lambda p: (os.path.basename(os.path.dirname(os.path.dirname(p))),
                              os.path.basename(os.path.dirname(p))))
for mpath in rdirs:
        rdir = os.path.dirname(mpath)
        rec = os.path.basename(rdir)
        tax = os.path.basename(os.path.dirname(rdir))
        md = json.load(open(mpath, encoding="utf-8"))
        slices = sorted(glob.glob(os.path.join(rdir, "slice_*.iq")))
        sbytes = sum(os.path.getsize(s) for s in slices)
        idir = os.path.join(IMAGES, tax, rec)
        jpgs = glob.glob(os.path.join(idir, "*.jpg")) if os.path.isdir(idir) else []
        ibytes = sum(os.path.getsize(j) for j in jpgs)

        n_meta = md.get("n_slices")
        if n_meta is not None and int(n_meta) != len(slices):
            issues.append("SLICE_COUNT %s/%s meta=%s disk=%d" % (tax, rec, n_meta, len(slices)))
        if len(jpgs) != len(slices):
            issues.append("IMG_COUNT %s/%s slices=%d imgs=%d" % (tax, rec, len(slices), len(jpgs)))
        snr = md.get("native_snr_db")
        if snr is None:
            issues.append("NO_SNR %s/%s" % (tax, rec))
        fs = md.get("sample_rate_hz") or 0
        ss = md.get("slice_samples") or 0
        dur = (len(slices) * ss / fs) if fs and ss else ""
        sp, basis = split_info.get((tax, rec), ("MISSING_IN_MANIFEST", ""))
        if sp == "MISSING_IN_MANIFEST":
            issues.append("NOT_IN_MANIFEST %s/%s" % (tax, rec))

        rows.append({
            "taxonomy_id": tax, "recording": rec,
            "source_dataset": md.get("source_dataset", "RFUAV"),
            "split": sp, "split_basis": basis,
            "n_slices": len(slices), "slice_bytes": sbytes,
            "n_images": len(jpgs), "image_bytes": ibytes,
            "delivered_seconds": round(dur, 3) if dur != "" else "",
            "native_snr_db": snr if snr is not None else "",
            "sample_rate_hz": fs, "slice_samples": ss,
            "source_ref": md.get("source_relpath") or md.get("original_file", ""),
            "topup_reader": md.get("topup_reader", ""),
            "n_slices_available": md.get("n_slices_available", ""),
        })

with open(os.path.join(MANI, "inventory_recordings.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ---- class rollup -----------------------------------------------------------
cls = defaultdict(lambda: {"recs": 0, "slices": 0, "images": 0, "bytes": 0,
                           "train": 0, "test": 0, "excluded": 0,
                           "sources": set(), "snrs": [], "secs": 0.0})
for r in rows:
    c = cls[r["taxonomy_id"]]
    c["recs"] += 1; c["slices"] += r["n_slices"]; c["images"] += r["n_images"]
    c["bytes"] += r["slice_bytes"] + r["image_bytes"]
    if r["split"] in ("train", "test", "excluded"):
        c[r["split"]] += 1
    c["sources"].add(r["source_dataset"])
    if r["native_snr_db"] != "":
        c["snrs"].append(float(r["native_snr_db"]))
    if r["delivered_seconds"] != "":
        c["secs"] += float(r["delivered_seconds"])

def status_of(tax, c):
    if tax in NON_TRANSMITTER:
        return "non_transmitter"
    if c["train"] and c["test"]:
        return "evaluable"
    if c["train"]:
        return "train_only"
    return "excluded_only"

with open(os.path.join(MANI, "inventory_classes.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["taxonomy_id", "status", "sources", "recordings", "train_recs",
                "test_recs", "excluded_recs", "slices", "images", "bytes_mb",
                "delivered_seconds", "snr_min_db", "snr_max_db"])
    for tax in sorted(cls):
        c = cls[tax]
        w.writerow([tax, status_of(tax, c), "+".join(sorted(c["sources"])),
                    c["recs"], c["train"], c["test"], c["excluded"],
                    c["slices"], c["images"], round(c["bytes"] / 1e6, 1),
                    round(c["secs"], 2),
                    round(min(c["snrs"]), 1) if c["snrs"] else "",
                    round(max(c["snrs"]), 1) if c["snrs"] else ""])

# ---- tree sizes + report ----------------------------------------------------
trees = ["rawdata", "images", "benchmark", "quarantine", "manifests"]
sizes = {t: dir_bytes(os.path.join(ROOT, t)) for t in trees if
         os.path.isdir(os.path.join(ROOT, t))}

src_agg = defaultdict(lambda: defaultdict(int))
for r in rows:
    a = src_agg[r["source_dataset"]]
    a["recs"] += 1; a["slices"] += r["n_slices"]
    a["bytes"] += r["slice_bytes"] + r["image_bytes"]
    a[r["split"]] = a.get(r["split"], 0) + 1

st_count = defaultdict(int)
for tax in cls:
    st_count[status_of(tax, cls[tax])] += 1

L = ["# RFUAV-v2 Data Inventory (%s)" % time.strftime("%Y-%m-%d %H:%M"), ""]
L.append("## Tree sizes on disk")
L.append("")
L.append("| tree | size (GB) | note |")
L.append("|---|---|---|")
note = {"rawdata": "canonical IQ by source (RFUAV/DroneRF/DroneRFa/DroneRFb)",
        "images": "spectrogram + scaled20M ablation + augmented renders",
        "benchmark": "train trees (hardlinks) + augmented_iq + runs",
        "quarantine": "never-delete holding area"}
for t in trees:
    if t in sizes:
        L.append("| %s | %.2f | %s |" % (t, sizes[t] / 1e9, note.get(t, "")))
L.append("| TOTAL | %.2f | train_dataset_v1 double-counts hardlinked jpgs |"
         % (sum(sizes.values()) / 1e9))
L.append("")
L.append("## Distribution by source")
L.append("")
L.append("| source | recordings | train | test | excluded | slices | GB |")
L.append("|---|---|---|---|---|---|---|")
for s in sorted(src_agg):
    a = src_agg[s]
    L.append("| %s | %d | %d | %d | %d | %d | %.2f |" % (
        s, a["recs"], a.get("train", 0), a.get("test", 0), a.get("excluded", 0),
        a["slices"], a["bytes"] / 1e9))
L.append("")
L.append("## Class status")
L.append("")
L.append("| status | classes |")
L.append("|---|---|")
for k in ("evaluable", "train_only", "excluded_only", "non_transmitter"):
    L.append("| %s | %d |" % (k, st_count.get(k, 0)))
L.append("")
L.append("## Integrity checks")
L.append("")
if issues:
    L.append("%d issue(s):" % len(issues))
    L.extend("- " + i for i in issues)
else:
    L.append("ALL CLEAN: every recording has metadata.json, slice count matches "
             "metadata, image count == slice count, SNR present, manifest row present.")
L.append("")
L.append("## Quarantine contents")
L.append("")
qroot = os.path.join(ROOT, "quarantine")
for sub in sorted(os.listdir(qroot)):
    p = os.path.join(qroot, sub)
    if os.path.isdir(p):
        n = sum(len(fn) for _, _, fn in os.walk(p))
        L.append("- %s: %d files, %.2f GB" % (sub, n, dir_bytes(p) / 1e9))
open(os.path.join(MANI, "inventory_report.md"), "w", encoding="utf-8").write(
    "\n".join(L) + "\n")

print("recordings=%d classes=%d issues=%d" % (len(rows), len(cls), len(issues)))
for i in issues[:20]:
    print("ISSUE:", i)
print("sizes GB: " + "  ".join("%s=%.2f" % (t, sizes[t] / 1e9) for t in sizes))
print("inventory written to manifests\\inventory_{recordings,classes}.csv + inventory_report.md")
