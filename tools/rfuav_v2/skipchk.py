# -*- coding: utf-8 -*-
"""Clinching test for the E-012 mechanism: if the slicer's filename counter
skipped an index, the skipped filename must be ABSENT from the directory."""
import csv, io, collections
M = r"E:\Drone_dataset\RFUAV-v2\manifests"
inv = [r["path"] for r in csv.DictReader(
    io.open(M + r"\e1_inventory.csv", "r", encoding="utf-8", newline=""))]
P = "E:\\Drone_dataset\\RFUAV\\"

targets = [("PublicDataset\\DJI AVATA2\\VTSBW=10", "pack1", 4),
           ("PublicDataset\\DJI FPV COMBO\\VTSBW=20", "pack3", 10),
           ("PublicDataset\\DJI MINI3\\VTSBW=10", "pack2", 8)]

for d, pack, skipped in targets:
    full = P + d
    files = sorted(p.split("\\")[-1] for p in inv
                   if p.startswith(full + "\\")
                   and p.split("\\")[-1].startswith(pack + "_"))
    print("\n=== %s [%s] : %d files ===" % (d, pack, len(files)))
    print("   " + ", ".join(files))
    want = "%s_%d-%ds.iq" % (pack, skipped, skipped + 1)
    print("   PREDICTED-ABSENT %-22s : %s" % (
        want, "ABSENT (mechanism confirmed)" if want not in files else "PRESENT (mechanism REFUTED)"))
    # is any other pack index in the same dir complete?
    others = collections.Counter(f.split("_")[0] for p in inv
                                 if p.startswith(full + "\\")
                                 for f in [p.split("\\")[-1]])
    print("   packs in dir: %s" % dict(others))
