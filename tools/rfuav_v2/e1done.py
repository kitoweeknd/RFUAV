# -*- coding: utf-8 -*-
"""Is E1 actually finished?  Compare SETS, not totals.

A 'percent by bytes' figure crossing 100 means the numerator and denominator
are drawn from different file sets, so it cannot answer the question.  This
diffs the hashed set against the inventory set path by path.
"""
import csv, io, os, collections

M = r"E:\Drone_dataset\RFUAV-v2\manifests"

hashed = {}
with io.open(M + r"\e1_hashes.csv", "r", encoding="utf-8", newline="") as f:
    for row in csv.reader(f):          # HEADERLESS: path,size,sha512
        if len(row) >= 3:
            try:
                hashed[row[0]] = int(row[1])
            except ValueError:
                pass

inv = {}
with io.open(M + r"\e1_inventory.csv", "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):      # HAS header: path,size,mtime_iso
        inv[row["path"]] = int(row["size"])

print("hashed rows      : %d  (%.1f GB)" % (len(hashed), sum(hashed.values()) / 2.0**30))
print("inventory rows   : %d  (%.1f GB)" % (len(inv), sum(inv.values()) / 2.0**30))

missing = {p: s for p, s in inv.items() if p not in hashed}
extra = {p: s for p, s in hashed.items() if p not in inv}
print("\nin inventory, NOT hashed : %d  (%.1f GB)"
      % (len(missing), sum(missing.values()) / 2.0**30))
print("hashed, NOT in inventory : %d  (%.1f GB)"
      % (len(extra), sum(extra.values()) / 2.0**30))

by_ext = collections.Counter()
by_ext_b = collections.Counter()
for p, s in missing.items():
    e = os.path.splitext(p)[1].lower()
    by_ext[e] += 1
    by_ext_b[e] += s
print("\nmissing by extension:")
for e, n in by_ext.most_common(15):
    print("  %-10s %5d files  %10.2f GB" % (e or "(none)", n, by_ext_b[e] / 2.0**30))

if extra:
    print("\nextra by extension:")
    ce = collections.Counter(os.path.splitext(p)[1].lower() for p in extra)
    for e, n in ce.most_common(10):
        print("  %-10s %5d" % (e or "(none)", n))

# size mismatches would mean the file changed under us -- must be zero
mism = [(p, inv[p], hashed[p]) for p in hashed if p in inv and hashed[p] != inv[p]]
print("\nsize mismatch inventory vs hashed: %d" % len(mism))
for p, a, b in mism[:5]:
    print("  %s  inv=%d hashed=%d" % (p[-70:], a, b))

SIG = (".iq", ".dat", ".bin", ".mat")
sig_missing = {p: s for p, s in missing.items() if p.lower().endswith(SIG)}
print("\nSIGNAL files still unhashed: %d  (%.2f GB)"
      % (len(sig_missing), sum(sig_missing.values()) / 2.0**30))
for p, s in sorted(sig_missing.items(), key=lambda kv: -kv[1])[:12]:
    print("  %10.3f GB  %s" % (s / 2.0**30, p[-80:]))
print("\nVERDICT: %s" % ("E1 COMPLETE for signal files"
                         if not sig_missing else "E1 STILL INCOMPLETE"))
