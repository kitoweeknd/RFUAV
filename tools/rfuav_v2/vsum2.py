# -*- coding: utf-8 -*-
import csv, io, collections
M = r"E:\Drone_dataset\RFUAV-v2\manifests"
sub = list(csv.DictReader(io.open(M + r"\draft_dedup_substring.csv", "r", encoding="utf-8", newline="")))
rec = list(csv.DictReader(io.open(M + r"\draft_recordings.csv", "r", encoding="utf-8", newline="")))
ano = list(csv.DictReader(io.open(M + r"\draft_anomalies.csv", "r", encoding="utf-8", newline="")))

print("=== ALL 12 BYTE-PROVEN E-012 FABRICATED WINDOWS ===")
sh = [r for r in sub if "offset_shift" in r["match_class"]]
sh.sort(key=lambda r: r["slice_relpath"])
for r in sh:
    d = float(r["derived_start_s"]); q = float(r["requested_start_s"])
    print("  claim %5.0fs -> REAL %5.0fs (%+.0fs)  %s  win=%s" % (
        q, d, d - q, r["match_class"].replace("_offset_shift", ""), r["verify_windows"]))
    print("     %s" % r["slice_relpath"])
    print("     src: %s" % r["source_relpath"])

print("\n=== do the shifts correlate with truncation? ===")
tr = set(a["relpath"] for a in ano if a["anomaly"] == "E-008_truncated_delivery")
n_sh_tr = sum(1 for r in sh if r["slice_relpath"] in tr)
print("  shifted slices that are ALSO truncated: %d / %d" % (n_sh_tr, len(sh)))

print("\n=== label accuracy of PublicDataset-style windows ===")
conf = sum(1 for r in sub if r["match_class"] in ("interior_confirms", "arith_unique_confirms"))
print("  claimed offset CONFIRMED by bytes : %d" % conf)
print("  claimed offset REFUTED by bytes   : %d" % len(sh))
print("  => label error rate %.2f%%" % (100.0 * len(sh) / (conf + len(sh))))

print("\n=== remaining orphans (4) ===")
for a in ano:
    if a["anomaly"] in ("slice_source_absent", "orphan_truncated_slice"):
        print("  [%s] %s" % (a["anomaly"], a["relpath"]))

print("\n=== the （局部） head-extract: did the interior probe find it? ===")
for r in sub:
    if "\u5c40\u90e8" in r["slice_relpath"]:
        print("  FOUND class=%s start=%ss windows=%s" % (
            r["match_class"], r["derived_start_s"], r["verify_windows"]))
        print("     src: %s" % r["source_relpath"])

print("\n=== biggest leakage groups (session -> n files, cross-root) ===")
g = collections.Counter()
roots = collections.defaultdict(set)
for r in rec:
    g[r["session_id"]] += 1
    roots[r["session_id"]].add(r["relpath"].split("\\")[0])
for s, n in g.most_common(8):
    print("  %-46s %3d files  roots=%d" % (s, n, len(roots[s])))
