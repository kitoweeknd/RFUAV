"""Find canonical RFUAV dirs with no split (orphans) and classify them."""
import os, re, json, io, csv
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
ASSIGN = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
SLICE_RE = re.compile(r"^slice_\d{4}\.iq$")

# assignment stems (uncut + all)
uncut_stems = set()
all_stems = {}
with io.open(ASSIGN, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        stem = os.path.splitext(os.path.basename(r["source_relpath"].replace("/", "\\")))[0]
        all_stems[stem] = (r.get("is_uncut"), r.get("is_slice"), r.get("split"))
        if r.get("is_uncut") == "1":
            uncut_stems.add(stem)

orphans = []
for tax in sorted(os.listdir(CANON)):
    tdir = os.path.join(CANON, tax)
    if not os.path.isdir(tdir) or tax.startswith("ext."):
        continue
    for rec in sorted(os.listdir(tdir)):
        rdir = os.path.join(tdir, rec)
        if not os.path.isdir(rdir):
            continue
        mpath = os.path.join(rdir, "metadata.json")
        has_meta = os.path.exists(mpath)
        src = None
        if has_meta:
            with open(mpath) as f:
                m = json.load(f)
            src = m.get("source_dataset")
            if src != "RFUAV" or m.get("split"):
                continue
        n_slices = sum(1 for f in os.listdir(rdir) if SLICE_RE.match(f))
        if has_meta and src == "RFUAV" and n_slices > 0:
            continue
        if not has_meta:
            files = os.listdir(rdir)
            in_assign = all_stems.get(rec, ("?", "?", "NOT_IN_ASSIGNMENT"))
            orphans.append((tax, rec, n_slices, len(files), in_assign))

print("orphan dirs (no metadata.json): %d" % len(orphans))
for tax, rec, ns, nf, ia in orphans:
    print("  %-28s %-48s slices=%d files=%d  assign(is_uncut,is_slice,split)=%s" % (tax[:28], rec[:48], ns, nf, ia))
