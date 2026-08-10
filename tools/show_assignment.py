import csv, io
from collections import Counter
m = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
with io.open(m, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
splits = Counter(r["split"] for r in rows)
roots = Counter(r["source_root"] for r in rows)
print("split distribution:")
for k, v in splits.most_common():
    print("  %s: %d" % (k, v))
print()
print("source_root distribution:")
for k, v in roots.most_common():
    print("  %s: %d" % (k, v))
print()
raw = [r for r in rows if r["source_root"] == "rawdata" and r["split"] == "train"]
print("rawdata train recordings: %d" % len(raw))
for r in raw[:3]:
    print("  %-30s  %s" % (r["taxonomy_id"], r["source_relpath"]))
print()
raw_test = [r for r in rows if r["source_root"] == "rawdata" and r["split"] == "test"]
print("rawdata test recordings: %d" % len(raw_test))
for r in raw_test[:3]:
    print("  %-30s  %s" % (r["taxonomy_id"], r["source_relpath"]))
print()
print("--- rawdata .iq files on disk ---")
print("location: E:\\Drone_dataset\\RFUAV\\rawdata\\<class>\\<iq_file>")
# show per-class counts
cls_counts = Counter()
for r in rows:
    if r["source_root"] == "rawdata" and r["split"] in ("train", "test"):
        cls_counts[(r["taxonomy_id"], r["split"])] += 1
for (c, s), n in sorted(cls_counts.items()):
    print("  %-8s  %-30s  %d recordings" % (s, c, n))
