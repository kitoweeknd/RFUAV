import os, io, csv, json
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
ASSIGN = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
for tax in ["rc.siyi.mk15", "rc.siyi.mk32", "rc.siyi.ft24"]:
    tdir = os.path.join(CANON, tax)
    print(tax + ":")
    for rec in sorted(os.listdir(tdir)):
        rdir = os.path.join(tdir, rec)
        if not os.path.isdir(rdir):
            continue
        has_meta = os.path.exists(os.path.join(rdir, "metadata.json"))
        n_iq = sum(1 for f in os.listdir(rdir) if f.endswith(".iq"))
        print("  %-52s meta=%s iq=%d" % (rec[:52], has_meta, n_iq))
print()
print("assignment rows for siyi classes:")
with io.open(ASSIGN, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if "siyi" in r["taxonomy_id"]:
            print("  %-14s uncut=%s slice=%s split=%-8s %s" % (
                r["taxonomy_id"], r["is_uncut"], r["is_slice"], r["split"],
                os.path.basename(r["source_relpath"].replace("/", "\\"))[:60]))
