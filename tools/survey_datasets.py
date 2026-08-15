"""Survey all datasets for SOTA class count claim."""
import os

# RFUAV
print("=== RFUAV (own) ===")
rd = r"E:\Drone_dataset\RFUAV\rawdata"
classes = [c for c in os.listdir(rd) if os.path.isdir(os.path.join(rd, c))]
total_iqs = 0
for c in sorted(classes):
    iqs = [f for f in os.listdir(os.path.join(rd, c)) if f.endswith(".iq")]
    total_iqs += len(iqs)
    print("  %-30s %d recordings" % (c, len(iqs)))
print("  TOTAL: %d classes, %d recordings" % (len(classes), total_iqs))

# DroneRF
print("\n=== DroneRF ===")
rf = r"E:\Drone_dataset\RF"
for cls in sorted(os.listdir(rf)):
    cp = os.path.join(rf, cls)
    if not os.path.isdir(cp):
        continue
    rars = [f for f in os.listdir(cp) if f.endswith(".rar")]
    print("  %-30s %d rar files" % (cls, len(rars)))

# DroneRFa
print("\n=== DroneRFa ===")
rfa = r"E:\Drone_dataset\RFA\DroneRFa-IQ"
for root, dirs, files in os.walk(rfa):
    dats = [f for f in files if f.endswith(".dat") and f != "RF0.dat"]
    if dats:
        rel = root.replace(rfa, "").strip("\\/")
        print("  %-30s %d .dat files" % (rel or "(top)", len(dats)))
        for f in sorted(dats)[:3]:
            sz = os.path.getsize(os.path.join(root, f)) / 1e6
            print("    %s (%.0f MB)" % (f, sz))

# DroneClip
print("\n=== DroneClip ===")
dc = r"E:\Drone_dataset\DroneClip-Dataset"
for root, dirs, files in os.walk(dc):
    rel = root.replace(dc, "").strip("\\/")
    if rel and dirs:
        continue
    if files:
        for f in sorted(files)[:3]:
            if f.endswith(".tsv"):
                print("  %s: %s" % (rel, f))
