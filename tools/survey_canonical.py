"""Survey canonical tree: per-class source, recordings, slices, SNR range."""
import os, json, re, csv
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
OUT = r"E:\Drone_dataset\RFUAV-v2\manifests\canonical_survey.csv"
SLICE_RE = re.compile(r"^slice_\d{4}\.iq$")

rows = []
for tax in sorted(os.listdir(CANON)):
    tdir = os.path.join(CANON, tax)
    if not os.path.isdir(tdir):
        continue
    for rec in sorted(os.listdir(tdir)):
        rdir = os.path.join(tdir, rec)
        if not os.path.isdir(rdir):
            continue
        files = os.listdir(rdir)
        n_slices = sum(1 for f in files if SLICE_RE.match(f))
        n_variants = sum(1 for f in files if f.endswith(".iq") and not SLICE_RE.match(f))
        meta = {}
        mpath = os.path.join(rdir, "metadata.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                meta = json.load(f)
        else:
            spath = os.path.join(rdir, "slice_0000_snr.json")
            if os.path.exists(spath):
                with open(spath) as f:
                    meta = {"native_snr_db": json.load(f).get("native_snr_db")}
        size_gb = sum(os.path.getsize(os.path.join(rdir, f)) for f in files) / 1e9
        rows.append({
            "taxonomy_id": tax, "recording": rec,
            "source_dataset": meta.get("source_dataset", "RFUAV"),
            "native_snr_db": meta.get("native_snr_db", ""),
            "sample_rate_hz": meta.get("sample_rate_hz", ""),
            "n_slices": n_slices, "n_variants": n_variants,
            "size_gb": round(size_gb, 2),
        })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Console summary by source
from collections import defaultdict
src = defaultdict(lambda: {"classes": set(), "recs": 0, "slices": 0, "gb": 0.0})
for r in rows:
    s = src[r["source_dataset"]]
    s["classes"].add(r["taxonomy_id"])
    s["recs"] += 1
    s["slices"] += r["n_slices"]
    s["gb"] += r["size_gb"]
print("%-15s %8s %6s %8s %10s" % ("source", "classes", "recs", "slices", "size_GB"))
for k in sorted(src):
    s = src[k]
    print("%-15s %8d %6d %8d %10.1f" % (k, len(s["classes"]), s["recs"], s["slices"], s["gb"]))
tot_c = len(set(r["taxonomy_id"] for r in rows))
print("TOTAL classes=%d recs=%d slices=%d size=%.1f GB" % (
    tot_c, len(rows), sum(r["n_slices"] for r in rows), sum(r["size_gb"] for r in rows)))
print("survey written: " + OUT)
