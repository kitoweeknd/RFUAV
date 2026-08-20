"""Build the augmented-data manifest: one row per augmented recording.

Covers E:\\Drone_dataset\\RFUAV-v2\\augmented\\{iq,images}. Records slice/image
counts, bytes, augmentation params coverage, and the source recording, so the
151 GB augmented IQ asset is fully accounted for.
"""
import os, csv, json, glob
from collections import defaultdict

ROOT = r"E:\Drone_dataset\RFUAV-v2"
AUG_IQ = os.path.join(ROOT, "benchmark", "augmented_iq")
AUG_IMG = os.path.join(ROOT, "images", "spectrogram_augmented")
OUT = os.path.join(ROOT, "manifests", "augmented_manifest_v1.csv")

rows = []
for tax in sorted(os.listdir(AUG_IQ)):
    tdir = os.path.join(AUG_IQ, tax)
    if not os.path.isdir(tdir):
        continue
    for rec in sorted(os.listdir(tdir)):
        rdir = os.path.join(tdir, rec)
        if not os.path.isdir(rdir):
            continue
        iqs = glob.glob(os.path.join(rdir, "slice_*.iq"))
        iq_bytes = sum(os.path.getsize(f) for f in iqs)
        md = {}
        mp = os.path.join(rdir, "metadata.json")
        if os.path.exists(mp):
            md = json.load(open(mp, encoding="utf-8"))
        img_dir = os.path.join(AUG_IMG, tax, rec)
        jpgs = glob.glob(os.path.join(img_dir, "*.jpg")) if os.path.isdir(img_dir) else []
        pfile = os.path.join(img_dir, "aug_params.jsonl")
        n_params = 0
        if os.path.exists(pfile):
            seen = set()
            for line in open(pfile, encoding="utf-8"):
                try:
                    seen.add(json.loads(line)["slice"])
                except Exception:
                    pass
            n_params = len(seen)
        rows.append({
            "taxonomy_id": tax,
            "aug_recording": rec,
            "source_recording": md.get("augmented_from", rec.replace("__aug1", "")),
            "augmentation": md.get("augmentation", "channel_sim_aug1"),
            "sample_rate_hz": md.get("sample_rate_hz", ""),
            "n_iq_slices": len(iqs),
            "iq_bytes": iq_bytes,
            "n_images": len(jpgs),
            "n_param_records": n_params,
            "reproducible_seed": "sha256(tax/rec/slice/aug1)",
        })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

n_iq = sum(r["n_iq_slices"] for r in rows)
n_img = sum(r["n_images"] for r in rows)
gb = sum(r["iq_bytes"] for r in rows) / 1e9
bad_param = [r for r in rows if r["n_param_records"] < r["n_iq_slices"]]
bad_img = [r for r in rows if r["n_images"] != r["n_iq_slices"]]
print("augmented manifest: %d recordings, iq_slices=%d (%.1f GB), images=%d"
      % (len(rows), n_iq, gb, n_img))
print("param coverage issues=%d, image/iq mismatch=%d" % (len(bad_param), len(bad_img)))
for r in (bad_param + bad_img)[:10]:
    print("  CHECK %s/%s iq=%d img=%d params=%d" % (
        r["taxonomy_id"], r["aug_recording"], r["n_iq_slices"],
        r["n_images"], r["n_param_records"]))
print("written: %s" % OUT)
