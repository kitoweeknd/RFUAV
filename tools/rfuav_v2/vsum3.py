# -*- coding: utf-8 -*-
import csv, io, collections
M = r"E:\Drone_dataset\RFUAV-v2\manifests"
sub = list(csv.DictReader(io.open(M + r"\draft_dedup_substring.csv", "r", encoding="utf-8", newline="")))
ano = list(csv.DictReader(io.open(M + r"\draft_anomalies.csv", "r", encoding="utf-8", newline="")))
print("anomalies cols: %s" % ",".join(ano[0].keys()))

# group every verified pair by its pack sequence (dir + pack index)
seqs = collections.defaultdict(list)
for r in sub:
    if str(r["verified_bytes"]) != "1":
        continue
    p = r["slice_relpath"]
    base = p.split("\\")[-1]
    d = "\\".join(p.split("\\")[:-1])
    pack = base.split("_")[0]
    try:
        q = float(r["requested_start_s"]); dv = float(r["derived_start_s"])
    except (TypeError, ValueError):
        continue
    seqs[(d, pack)].append((q, dv, base))

print("\n=== SEQUENCES CONTAINING ANY SHIFT: where does the desync start? ===")
for k in sorted(seqs):
    rows = sorted(seqs[k])
    if not any(abs(dv - q) > 1e-6 for q, dv, _ in rows):
        continue
    print("\n  %s  [%s]  (%d verified slices)" % (k[0], k[1], len(rows)))
    prev = None
    for q, dv, base in rows:
        off = dv - q
        mark = "   OK" if abs(off) < 1e-6 else "  <-- SHIFT %+.0fs" % off
        if prev is not None and abs(off - prev) > 1e-6:
            mark += "   *** DESYNC BOUNDARY ***"
        print("     claim %5.0fs -> real %5.0fs%s" % (q, dv, mark))
        prev = off

print("\n=== shift vs truncation ===")
kcol = "relpath" if "relpath" in ano[0] else list(ano[0].keys())[0]
acol = [c for c in ano[0] if "anom" in c.lower() or "code" in c.lower() or "kind" in c.lower()]
acol = acol[0] if acol else list(ano[0].keys())[1]
print("  using key=%s anomaly=%s" % (kcol, acol))
tr = set(a[kcol] for a in ano if "E-008" in a[acol])
sh = [r for r in sub if "offset_shift" in r["match_class"]]
print("  shifted slices also truncated: %d / %d" % (
    sum(1 for r in sh if r["slice_relpath"] in tr), len(sh)))

print("\n=== remaining orphans ===")
for a in ano:
    if "absent" in a[acol] or "orphan" in a[acol]:
        print("  [%s] %s" % (a[acol], a[kcol]))

print("\n=== the （局部） head extract ===")
for r in sub:
    if "\u5c40\u90e8" in r["slice_relpath"]:
        print("  class=%s start=%ss win=%s verified=%s" % (
            r["match_class"], r["derived_start_s"], r["verify_windows"], r["verified_bytes"]))
        print("     src: %s" % r["source_relpath"])
