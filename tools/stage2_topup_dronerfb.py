"""Stage 2: top up DroneRFb canonical sessions from 3 to up to 10 slices.

Rebuilds the same per-session sorted slice lists as the two import rounds
(train/ dir for individuals 1-2 + background; test_labels.txt mapping for
individuals 3), then appends slices [n_current:10] in identical sort order.
"""
import os, re, json, time
import numpy as np
import h5py

BASE = r"E:\Drone_dataset\DroneRFb\twin_droneRF"
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
TARGET = 10

def log(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

def read_iq(fpath):
    with h5py.File(fpath, "r") as hf:
        I = np.array(hf["I"]).ravel().astype(np.float32)
        Q = np.array(hf["Q"]).ravel().astype(np.float32)
    n = min(len(I), len(Q))
    iq = np.empty(n, dtype=np.complex64)
    iq.real = I[:n]; iq.imag = Q[:n]
    return iq

sessions = {}  # key -> sorted [(slice_n, disk_path, orig_name)]

# train/ : individuals 1-2
train_dir = os.path.join(BASE, "train")
pat_train = re.compile(r"^([A-G][12])_(IN|OUT)(?:_(S\d+))?_slice_(\d+)\.mat$")
pat_bg = re.compile(r"^background_slice_(\d+)\.mat$")
for f in os.listdir(train_dir):
    m = pat_train.match(f)
    if m:
        key = (m.group(1), m.group(2), m.group(3) or "S0")
        sessions.setdefault(key, []).append((int(m.group(4)), os.path.join(train_dir, f), f))
        continue
    m = pat_bg.match(f)
    if m:
        sessions.setdefault(("background", "NA", "S0"), []).append(
            (int(m.group(1)), os.path.join(train_dir, f), f))

# test/ : individuals 3 only (1-2 test slices belong to DroneRFb's own held-out
# protocol; keep canonical sessions single-origin)
test_dir = os.path.join(BASE, "test")
pat_test = re.compile(r"^([A-G]3)_(IN|OUT)(?:_(S\d+))?_slice_(\d+)\.mat$")
with open(os.path.join(BASE, "test_labels.txt"), encoding="utf-8", errors="replace") as f:
    for line in f:
        parts = line.split()
        if len(parts) != 2:
            continue
        m = pat_test.match(parts[0])
        if not m:
            continue
        dpath = os.path.join(test_dir, parts[1] + ".mat")
        if not os.path.exists(dpath):
            continue
        key = (m.group(1), m.group(2), m.group(3) or "S0")
        sessions.setdefault(key, []).append((int(m.group(4)), dpath, parts[0]))

log("Stage 2 start: %d DroneRFb sessions rebuilt" % len(sessions))

topped = unchanged = 0
added_gb = 0.0
for key in sorted(sessions):
    code, inout, scene = key
    tax_id = "ext.dronerfb." + code.lower()
    rec_stem = "drfb_%s" % code if inout == "NA" else "drfb_%s_%s_%s" % (code, inout, scene)
    rec_dir = os.path.join(CANON, tax_id, rec_stem)
    meta_path = os.path.join(rec_dir, "metadata.json")
    if not os.path.exists(meta_path):
        log("  NO CANONICAL REC for %s (unexpected)" % rec_stem)
        continue

    slist = sorted(sessions[key])
    n_current = sum(1 for f in os.listdir(rec_dir)
                    if re.match(r"^slice_\d{4}\.iq$", f))
    want = min(TARGET, len(slist))
    if n_current >= want:
        unchanged += 1
        continue

    appended = []
    wi = n_current
    for slice_n, dpath, oname in slist[n_current:want]:
        try:
            iq = read_iq(dpath)
        except Exception as e:
            log("  READ FAIL %s: %s" % (dpath, e))
            continue
        if iq.size == 0 or np.max(np.abs(iq)) < 1e-15:
            continue
        data = np.empty(2 * iq.size, dtype=np.float32)
        data[0::2] = iq.real; data[1::2] = iq.imag
        out = os.path.join(rec_dir, "slice_%04d.iq" % wi)
        data.tofile(out)
        added_gb += os.path.getsize(out) / 1e9
        appended.append([os.path.basename(dpath), oname])
        wi += 1

    with open(meta_path) as f:
        meta = json.load(f)
    meta["n_slices"] = wi
    meta["n_slices_available"] = len(slist)
    meta.setdefault("topup_source_files", []).extend(appended)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    topped += 1
    log("  %-24s %d -> %d slices" % (rec_stem, n_current, wi))

log("Stage 2 DONE: topped=%d unchanged=%d added=%.1f GB" % (topped, unchanged, added_gb))
