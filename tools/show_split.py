import csv, io
from collections import Counter
m = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
with io.open(m, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

splits = Counter(r["split"] for r in rows)
print("=== Split distribution ===")
for k, v in splits.most_common():
    print("  %s: %d" % (k, v))
print("  Total: %d" % sum(splits.values()))

print()
print("=== Per-class train/test ===")
for cls in sorted(set(r["taxonomy_id"] for r in rows)):
    c = Counter(r["split"] for r in rows if r["taxonomy_id"] == cls)
    train = c.get("train", 0)
    test = c.get("test", 0)
    excl = c.get("excluded", 0)
    in_both = "BOTH" if train > 0 and test > 0 else ""
    print("  %-35s  train=%-3d  test=%-3d  excluded=%-3d  %s" % (cls, train, test, excl, in_both))

print()
print("=== Summary ===")
train_cls = set(r["taxonomy_id"] for r in rows if r["split"] == "train")
test_cls = set(r["taxonomy_id"] for r in rows if r["split"] == "test")
excl_cls = set(r["taxonomy_id"] for r in rows if r["split"] == "excluded")
both = train_cls & test_cls
train_only = train_cls - test_cls
print("train+test both: %d classes" % len(both))
print("train-only: %d classes" % len(train_only))
print("test-only: %d classes" % len(test_cls - train_cls))
print("excluded: %d classes" % len(excl_cls))
