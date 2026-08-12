"""SNR-stratified evaluation: measure SNR of each val frame, then group by SNR level."""
import sys, os, numpy as np, torch, timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict

sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
# We'll estimate SNR from the spectrogram image itself using a simpler approach
# Actually, we need the original .iq to measure SNR. But we already have canonical slices.
# For speed, let's just use the existing split_23 val and measure SNR from the source .iq
# using the canonical slice data we already have.

# Alternative: use the SNR sidecar JSON from canonicalization
# Canonical tree: canonical/<class>/<recording>/slice_0000_snr.json
# We already measured SNR for all 254 recordings!

import csv, io, json

m = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
canon = r"E:\Drone_dataset\RFUAV-v2\canonical"

# Build recording -> native SNR map
snr_map = {}
for cls in os.listdir(canon):
    cp = os.path.join(canon, cls)
    if not os.path.isdir(cp): continue
    for rec in os.listdir(cp):
        rp = os.path.join(cp, rec)
        if not os.path.isdir(rp): continue
        snr_file = os.path.join(rp, "slice_0000_snr.json")
        if os.path.exists(snr_file):
            with open(snr_file) as f:
                snr_map[rec] = json.load(f)["native_snr_db"]

# Now map each val image to its recording's SNR
# The val directory structure: split_23/val/<class>/<image_name>
# Image names contain the recording stem: e.g. "DEVENTION_DEVO_10-daifei-60db_2440m_100m(2)_rc.devention.devo_10 (0).jpg"
val_path = r"E:\Drone_dataset\RFUAV-v2\split_23\val"

# Build image -> SNR mapping
image_snr = {}
for cls in os.listdir(val_path):
    cp = os.path.join(val_path, cls)
    if not os.path.isdir(cp): continue
    for fname in os.listdir(cp):
        if not fname.endswith(".jpg"): continue
        # Extract recording stem from filename
        # Format: <recording_stem>_<class> (frame).jpg
        # The recording stem is everything before the last "_<class>"
        parts = fname.rsplit("_" + cls, 1)
        if len(parts) == 2:
            stem = parts[0]
            # Match against snr_map keys
            for rec_key, snr in snr_map.items():
                if rec_key in stem or stem in rec_key:
                    image_snr[os.path.join(cls, fname)] = snr
                    break
            else:
                # Try fuzzy match
                image_snr[os.path.join(cls, fname)] = None

print("Images with SNR: %d / %d" % (sum(1 for v in image_snr.values() if v is not None), len(image_snr)))

# SNR bins
bins = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 100)]
for lo, hi in bins:
    imgs = [k for k, v in image_snr.items() if v is not None and lo <= v < hi]
    print("  SNR [%d-%d): %d images" % (lo, hi, len(imgs)))
