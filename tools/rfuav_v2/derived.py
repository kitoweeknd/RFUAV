# -*- coding: utf-8 -*-
"""Profile the DERIVED artefact tree from the inventory alone (no disk reads).

Why this matters and is not a detour: 796.6 GB of .jpg is not raw data, it is
spectrogram frames rendered FROM the recordings.  If a split is drawn over those
frames, it inherits every bit of the recording-identity leakage the substring
work just documented -- and it inherits it invisibly, because at the frame level
the provenance is gone from the path.  So before E2 designs the canonical tree,
the question is: do the frame paths still carry enough provenance to group them
back onto a recording?  If yes, grouped CV is possible.  If no, these frames are
unusable for any leakage-clean evaluation and that must be stated.
"""
import csv, io, os, collections, re

M = r"E:\Drone_dataset\RFUAV-v2\manifests"

jpg_dirs = collections.Counter()
jpg_bytes = collections.Counter()
pth = []
top = collections.Counter()
top_b = collections.Counter()
samples = collections.defaultdict(list)

with io.open(M + r"\e1_inventory.csv", "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        p = row["path"]; s = int(row["size"])
        e = os.path.splitext(p)[1].lower()
        rel = p.split("Drone_dataset\\", 1)[-1]
        parts = rel.split("\\")
        t = "\\".join(parts[:2]) if len(parts) > 1 else rel
        top[t] += 1; top_b[t] += s
        if e == ".jpg":
            d = os.path.dirname(rel)
            jpg_dirs[d] += 1; jpg_bytes[d] += s
            if len(samples[t]) < 3:
                samples[t].append(rel)
        elif e == ".pth":
            pth.append((s, rel))

G = 2.0 ** 30
print("=== top-2-level dirs by BYTES ===")
for t, b in sorted(top_b.items(), key=lambda kv: -kv[1])[:18]:
    print("  %9.1f GB  %8d files  %s" % (b / G, top[t], t))

print("\n=== .jpg: %d distinct directories ===" % len(jpg_dirs))
print("largest frame directories:")
for d, n in jpg_dirs.most_common(12):
    print("  %7d frames  %8.2f GB  %s" % (n, jpg_bytes[d] / G, d[:95]))

print("\n=== sample frame paths (provenance check) ===")
for t in list(samples)[:8]:
    for rel in samples[t][:2]:
        print("  " + rel[:130])

print("\n=== .pth checkpoints: %d files, %.1f GB ===" % (len(pth), sum(x[0] for x in pth) / G))
for s, rel in sorted(pth, reverse=True)[:8]:
    print("  %7.2f GB  %s" % (s / G, rel[:110]))
