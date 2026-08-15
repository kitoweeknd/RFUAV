import os, json
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
classes = sorted(os.listdir(CANON))
print("ALL CLASSES (%d):" % len(classes))
for c in classes:
    print("  " + c)
print()
# Inspect 3 RFUAV recording dirs (no source_dataset in metadata => RFUAV)
shown = 0
for tax in classes:
    tdir = os.path.join(CANON, tax)
    if not os.path.isdir(tdir) or tax.startswith("ext."):
        continue
    for rec in sorted(os.listdir(tdir)):
        rdir = os.path.join(tdir, rec)
        if not os.path.isdir(rdir):
            continue
        files = sorted(os.listdir(rdir))
        mpath = os.path.join(rdir, "metadata.json")
        has_meta = os.path.exists(mpath)
        if not has_meta and shown < 3:
            print("RFUAV-style rec: %s/%s  files=%s" % (tax, rec, files[:8]))
            shown += 1
    if shown >= 3:
        break
# DroneRFa sample metadata
for tax in classes:
    tdir = os.path.join(CANON, tax)
    if not os.path.isdir(tdir):
        continue
    for rec in sorted(os.listdir(tdir)):
        mpath = os.path.join(tdir, rec, "metadata.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                m = json.load(f)
            if m.get("source_dataset") == "DroneRFa":
                print("\nDroneRFa sample metadata (%s/%s):" % (tax, rec))
                print(json.dumps(m, indent=2)[:800])
                raise SystemExit
