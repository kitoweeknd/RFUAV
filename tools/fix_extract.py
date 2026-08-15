"""Fix: extract merged zip into canonical directly."""
import os, sys, json, time, zipfile, numpy as np

sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

DST = r"E:\Drone_dataset\DroneRFb"
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
EXTRACTED = os.path.join(DST, "extracted")
MERGED = os.path.join(DST, "merged_full.zip")

# Try extracting the merged zip
print("Testing merged zip...")
try:
    with zipfile.ZipFile(MERGED, "r") as zf:
        names = zf.namelist()[:20]
        print("Files in zip:", len(names) if names else 0)
        for n in names[:10]:
            print(" ", n)
        if names:
            # Extract a few test files
            for n in names[:5]:
                try:
                    zf.extract(n, EXTRACTED)
                    print("  Extracted:", n)
                except Exception as e:
                    print("  Extract error:", e)
except Exception as e:
    print("Zip open error:", e)
    # Try reading raw bytes and checking magic
    with open(MERGED, "rb") as f:
        magic = f.read(4)
        print("First 4 bytes:", magic.hex())
        # Check if it's actually a zip
        if magic[:2] == b"PK":
            print("OK: valid zip header")
        else:
            print("NOT a zip file!")
