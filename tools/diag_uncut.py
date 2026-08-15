import os, io, csv
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
ASSIGN = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
missing = []
n_uncut = 0
with io.open(ASSIGN, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r.get("is_uncut") != "1":
            continue
        n_uncut += 1
        stem = os.path.splitext(os.path.basename(r["source_relpath"].replace("/", "\\")))[0]
        rdir = os.path.join(CANON, r["taxonomy_id"], stem)
        if not os.path.exists(os.path.join(rdir, "metadata.json")):
            missing.append((r["taxonomy_id"], stem, r.get("source_root"), r.get("split"), r["source_relpath"]))
print("total uncut rows: %d   missing from canonical: %d" % (n_uncut, len(missing)))
for t, s, root, sp, rel in missing:
    src = os.path.normpath(os.path.join(r"E:\Drone_dataset\RFUAV", rel))
    print("  %-22s %-52s root=%-12s split=%-8s exists=%s" % (t, s[:52], root, sp, os.path.exists(src)))

# sessions without any uncut member
sess = {}
with io.open(ASSIGN, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r.get("split") not in ("train", "test", "excluded"):
            continue
        sid = r["session_id"]
        d = sess.setdefault(sid, {"has_uncut": False, "slices": 0, "tax": r["taxonomy_id"],
                                  "splits": set(), "roots": set()})
        d["splits"].add(r["split"])
        d["roots"].add(r.get("source_root"))
        if r.get("is_uncut") == "1":
            d["has_uncut"] = True
        if r.get("is_slice") == "1":
            d["slices"] += 1
no_uncut = {k: v for k, v in sess.items() if not v["has_uncut"] and v["slices"] > 0}
print("\nsessions total=%d  slice-only=%d" % (len(sess), len(no_uncut)))
from collections import Counter
c = Counter(v["tax"] for v in no_uncut.values())
for tax, n in sorted(c.items()):
    print("  %-26s %d slice-only sessions" % (tax, n))
mixed = [k for k, v in no_uncut.items() if len(v["splits"]) > 1]
print("slice-only sessions with MIXED splits: %d %s" % (len(mixed), mixed[:5]))
roots = Counter(tuple(sorted(v["roots"])) for v in no_uncut.values())
print("roots distribution:", dict(roots))
