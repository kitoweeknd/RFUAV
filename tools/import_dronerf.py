"""Import the DroneRF dataset (Al-Sa'd et al.) into canonical.

Source: E:\\Drone_dataset\\RF\\<class dir>\\RF Data_<BUI>_<band>.rar, each
holding ~21 CSV text segments of 10M real samples @ 40 MS/s (0.25 s each).
L/H = two USRPs covering the lower/upper 40 MHz halves of the 2.4 GHz band,
recorded simultaneously -> same session (split must keep them together).

Per rar -> one canonical recording dronerf_<BUI>_<band>:
  concat segments in order, cut 0.1 s slices (4M real samples), convert
  real -> complex64 via analytic signal (hilbert) + (-10 MHz) shift so the
  0..20 MHz content is centered, write float32 interleaved IQ, FCME SNR.
Streaming: extract one rar -> convert -> delete the text (disk-safe).
Originals READ-ONLY; writes only under RFUAV-v2.
"""
import os, re, sys, json, glob, time, shutil, subprocess
import numpy as np

sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from scipy.signal import hilbert
from snr_dual import FCMEEstimator

RF_ROOT = r"E:\Drone_dataset\RF"
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
STAGING = r"E:\Drone_dataset\RFUAV-v2\staging\dronerf"
SEVENZ = r"C:\Program Files\7-Zip\7z.exe"
CLASS_MAP = {
    "Bepop drone": "uav.parrot.bebop",
    "AR drone": "uav.parrot.ar_drone",
    "Phantom drone": "uav.dji.phantom3",
    "Background RF activites": "bg.dronerf",
}
FS = 40e6
SS = 4_000_000        # 0.1 s of real samples
TARGET = 20
RAR_RE = re.compile(r"^RF Data_(\d+)_([HL])\.rar$", re.IGNORECASE)


def log(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


shift = np.exp(-1j * 2 * np.pi * 10e6 * np.arange(SS) / FS).astype(np.complex64)

done = skipped = failed = 0
for cdir, tax in sorted(CLASS_MAP.items()):
    src_dir = os.path.join(RF_ROOT, cdir)
    if not os.path.isdir(src_dir):
        log("MISSING DIR %s" % src_dir)
        continue
    for rar in sorted(os.listdir(src_dir)):
        m = RAR_RE.match(rar)
        if not m:
            continue
        bui, band = m.group(1), m.group(2).upper()
        rec = "dronerf_%s_%s" % (bui, band)
        rdir = os.path.join(CANON, tax, rec)
        if os.path.exists(os.path.join(rdir, "metadata.json")):
            skipped += 1
            continue
        # extract
        if os.path.isdir(STAGING):
            shutil.rmtree(STAGING)
        os.makedirs(STAGING, exist_ok=True)
        rc = subprocess.call([SEVENZ, "x", "-y", "-o" + STAGING,
                              os.path.join(src_dir, rar)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc != 0:
            log("EXTRACT FAIL %s (rc=%d)" % (rar, rc))
            failed += 1
            continue
        csvs = glob.glob(os.path.join(STAGING, "**", "*.csv"), recursive=True)

        def seg_idx(p):
            mm = re.search(r"_(\d+)\.csv$", os.path.basename(p))
            return int(mm.group(1)) if mm else 1 << 30

        csvs.sort(key=seg_idx)
        if not csvs:
            log("NO CSV in %s" % rar)
            failed += 1
            continue

        os.makedirs(rdir, exist_ok=True)
        buf = np.empty(0, dtype=np.float32)
        n_slices = 0
        total_real = 0
        bytes_per_samp = None
        for cpath in csvs:
            if n_slices >= TARGET and bytes_per_samp is not None:
                total_real += int(os.path.getsize(cpath) / bytes_per_samp)
                continue
            x = np.fromfile(cpath, sep=",", dtype=np.float64).astype(np.float32)
            if bytes_per_samp is None and len(x):
                bytes_per_samp = os.path.getsize(cpath) / len(x)
            total_real += len(x)
            buf = np.concatenate([buf, x]) if len(buf) else x
            while len(buf) >= SS and n_slices < TARGET:
                seg = buf[:SS]
                buf = buf[SS:]
                y = hilbert(seg.astype(np.float64)).astype(np.complex64)
                y = (y * shift).astype(np.complex64)
                out = np.empty(2 * SS, dtype=np.float32)
                out[0::2] = y.real
                out[1::2] = y.imag
                out.tofile(os.path.join(rdir, "slice_%04d.iq" % n_slices))
                n_slices += 1

        if n_slices == 0:
            log("FAIL %s: no slices produced" % rec)
            shutil.rmtree(rdir, ignore_errors=True)
            failed += 1
            continue

        # SNR from slice 0
        raw = np.fromfile(os.path.join(rdir, "slice_0000.iq"),
                          dtype=np.float32, count=2 * SS)
        iq = np.empty(SS, dtype=np.complex64)
        iq.real = raw[0::2]
        iq.imag = raw[1::2]
        snr, diag = FCMEEstimator(fs=FS, nfft=16384).estimate(iq)

        meta = {
            "source_dataset": "DroneRF",
            "taxonomy_id": tax,
            "session_id": "dronerf_bui_%s" % bui,
            "mode_bui": bui,
            "band": band,
            "native_snr_db": round(float(snr), 1) if np.isfinite(snr) else None,
            "sample_rate_hz": FS,
            "slice_samples": SS,
            "n_slices": n_slices,
            "n_slices_available": int(total_real // SS),
            "original_format": "csv_text_real_40MSps",
            "real_to_complex": "hilbert analytic + (-10 MHz) shift, complex64",
            "source_relpath": "RF\\%s\\%s" % (cdir, rar),
        }
        json.dump(meta, open(os.path.join(rdir, "metadata.json"), "w",
                             encoding="utf-8"), indent=2)
        done += 1
        log("OK %-22s %-18s slices=%d/%d SNR=%s dB" % (
            tax[:22], rec, n_slices, meta["n_slices_available"],
            meta["native_snr_db"]))

if os.path.isdir(STAGING):
    shutil.rmtree(STAGING)
log("DroneRF import done: recs=%d skipped=%d failed=%d" % (done, skipped, failed))
if failed:
    sys.exit(1)
