# -*- coding: utf-8 -*-
"""Break byte-verification down by match_class (correct column names)."""
import csv, io, collections
P = r"E:\Drone_dataset\RFUAV-v2\manifests\draft_dedup_substring.csv"
rows = list(csv.DictReader(io.open(P, "r", encoding="utf-8", newline="")))

def T(r):
    return str(r.get("verified_bytes", "")).strip() in ("1", "True", "true")

print("=== E-012 FABRICATED WINDOWS (byte-proven) ===")
for r in rows:
    if r["match_class"] == "arith_unique_offset_shift":
        print("  verified=%s windows=%s  claimed_start=%ss  DERIVED_start=%ss  cross_root=%s" % (
            r["verified_bytes"], r["verify_windows"], r["requested_start_s"],
            r["derived_start_s"], r["cross_root"]))
        print("     slice : %s  (%s B)" % (r["slice_relpath"], r["slice_size"]))
        print("     source: %s  (%s B)" % (r["source_relpath"], r["source_size"]))
        print("     offset: %s B  rate=%s MS/s  bps=%s" % (
            r["offset_bytes"], r["rate_msps"], r["bytes_per_sample"]))

print("\n=== uninformative_round THAT SURVIVED BYTES (%d) ===" % sum(
    1 for r in rows if r["match_class"] == "uninformative_round" and T(r)))
for r in rows:
    if r["match_class"] == "uninformative_round" and T(r):
        print("  start=%ss cross_root=%s" % (r["derived_start_s"], r["cross_root"]))
        print("     slice : %s" % r["slice_relpath"])
        print("     source: %s" % r["source_relpath"])

print("\n=== arith_unique_confirms: all %d verified? %s ===" % (
    sum(1 for r in rows if r["match_class"] == "arith_unique_confirms"),
    all(T(r) for r in rows if r["match_class"] == "arith_unique_confirms")))
cr = collections.Counter()
for r in rows:
    if r["match_class"] == "arith_unique_confirms":
        cr[r["slice_relpath"].split("\\")[0]] += 1
for k, v in cr.most_common():
    print("  %-30s %d" % (k, v))

print("\n=== REFUTED (88) : how far off? sample 8 ===")
n = 0
for r in rows:
    if not T(r):
        n += 1
        if n <= 8:
            print("  claimed=%ss derived=%ss" % (r["requested_start_s"], r["derived_start_s"]))
            print("     slice : %s" % r["slice_relpath"])
            print("     source: %s" % r["source_relpath"])
print("  total refuted: %d" % n)
