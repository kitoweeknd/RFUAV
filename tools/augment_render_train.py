"""Channel-augment every TRAIN-side slice of the evaluable classes, KEEPING
the augmented raw IQ (float32 interleaved) alongside the rendered image.

- valid/test slices are NEVER touched; canonical is read-only.
- Params drawn per-slice from a deterministic seed (sha256 of tax/rec/slice)
  -> fully reproducible; recorded in aug_params.jsonl per recording.
- complex64 kept throughout (pipeline invariant).
- Aborts if E: free space drops under 30 GB.
Output: E:\\Drone_dataset\\RFUAV-v2\\augmented\\iq\\<tax>\\<rec>__aug1\\slice_NNNN.iq
        E:\\Drone_dataset\\RFUAV-v2\\augmented\\images\\<tax>\\<rec>__aug1\\..._sNNNN.jpg
"""
import os, sys, csv, json, glob, time, hashlib, shutil
import numpy as np

sys.path.insert(0, r"D:\ML\RFUAV")
from graphic.RawDataProcessor import generate_images
from ChannelSimulation.channels.fading import apply_rayleigh_or_rician_fading
from ChannelSimulation.channels.doppler import apply_doppler
from ChannelSimulation.channels.multipath import apply_multipath
from ChannelSimulation.channels.cfo import apply_cfo
from ChannelSimulation.channels.phase_noise import apply_phase_noise
from ChannelSimulation.channels.iq_imbalance import apply_iq_imbalance
from ChannelSimulation.channels.awgn import apply_awgn

ROOT = r"E:\Drone_dataset\RFUAV-v2"
RAW = os.path.join(ROOT, "rawdata")
MANIFEST = os.path.join(ROOT, "manifests", "canonical_manifest_v1.csv")
AUG_IMG = os.path.join(ROOT, "images", "spectrogram_augmented")
AUG_IQ = os.path.join(ROOT, "benchmark", "augmented_iq")
MIN_FREE_GB = 30
NON_TRANSMITTER = {"bg.kong", "ext.dronerfb.background", "unk.weizhi"}


def log(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
by_class = {}
for r in rows:
    by_class.setdefault(r["taxonomy_id"], set()).add(r["split"])
eligible = {t for t, s in by_class.items()
            if t not in NON_TRANSMITTER and "train" in s and "test" in s}
SRC_DIR = {"RFUAV": "RFUAV", "DroneRF": "DroneRF", "DroneRFa": "DroneRFa", "DroneRFb-DIR": "DroneRFb"}
train_recs = [(r["taxonomy_id"], r["recording"], float(r["sample_rate_hz"]), SRC_DIR[r["source_dataset"]])
              for r in rows
              if r["taxonomy_id"] in eligible and r["split"] == "train"]
log("train recordings to augment: %d (classes=%d)" % (len(train_recs), len(eligible)))


def rng_for(tax, rec, si):
    h = hashlib.sha256(("%s/%s/%d/aug1" % (tax, rec, si)).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def augment(iq, fs, rng):
    """Random channel chain in canonical stage order. Returns (y, params)."""
    p = {}
    y = iq
    if rng.random() < 0.8:
        ftype = "rayleigh" if rng.random() < 0.5 else "rician"
        k = float(rng.uniform(2.0, 10.0))
        y = apply_rayleigh_or_rician_fading(y, fading_type=ftype, k_factor=k,
                                            block_size=256, rng=rng)
        p["fading"] = {"type": ftype, "k": round(k, 2)}
    if rng.random() < 0.7:
        sh = float(rng.uniform(20, 300)) * (1 if rng.random() < 0.5 else -1)
        sp = float(rng.uniform(10, 150))
        y = apply_doppler(y, fs, doppler_shift_hz=sh, doppler_spread_hz=sp, rng=rng)
        p["doppler"] = {"shift_hz": round(sh, 1), "spread_hz": round(sp, 1)}
    if rng.random() < 0.5:
        d = int(rng.integers(5, 41)); npth = int(rng.integers(2, 7))
        dec = float(rng.uniform(0.2, 0.6))
        y = apply_multipath(y, max_delay_samp=d, n_paths=npth, exp_decay=dec, rng=rng)
        p["multipath"] = {"max_delay": d, "n_paths": npth, "decay": round(dec, 2)}
    if rng.random() < 0.9:
        c = float(rng.uniform(500, 20000)) * (1 if rng.random() < 0.5 else -1)
        y = apply_cfo(y, fs, cfo_hz=c)
        p["cfo_hz"] = round(c, 1)
    if rng.random() < 0.7:
        s = float(rng.uniform(5e-4, 4e-3))
        y = apply_phase_noise(y, phase_std_per_sample=s, rng=rng)
        p["phase_std"] = round(s, 5)
    if rng.random() < 0.5:
        g = float(rng.uniform(0.2, 1.5)); ph = float(rng.uniform(0.5, 4.0))
        y = apply_iq_imbalance(y, gain_imbalance_db=g, phase_imbalance_deg=ph)
        p["iq_imb"] = {"gain_db": round(g, 2), "phase_deg": round(ph, 2)}
    if rng.random() < 0.8:
        snr = float(rng.uniform(5, 25))
        y = apply_awgn(y, snr_db=snr, rng=rng)
        p["awgn_snr_db"] = round(snr, 1)
    if not p:  # never emit an unmodified copy
        c = float(rng.uniform(2000, 20000))
        y = apply_awgn(apply_cfo(y, fs, cfo_hz=c), snr_db=15.0, rng=rng)
        p["cfo_hz"] = round(c, 1); p["awgn_snr_db"] = 15.0
    # keep render dynamic range comparable to the original
    rms_in = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    rms_out = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    if rms_out > 0:
        y = (y * np.float32(rms_in / rms_out)).astype(np.complex64)
    return y, p


done = skipped = failed = iq_bytes = 0
t0 = time.time()
for tax, rec, fs, srcd in train_recs:
    rdir = os.path.join(RAW, srcd, tax, rec)
    slices = sorted(glob.glob(os.path.join(rdir, "slice_*.iq")))
    aug_rec = rec + "__aug1"
    out_dir = os.path.join(AUG_IMG, tax, aug_rec)
    iq_dir = os.path.join(AUG_IQ, tax, aug_rec)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(iq_dir, exist_ok=True)
    json.dump({"sample_rate_hz": fs, "taxonomy_id": tax, "augmented_from": rec,
               "augmentation": "channel_sim_aug1"},
              open(os.path.join(iq_dir, "metadata.json"), "w"), indent=2)
    pfile = os.path.join(out_dir, "aug_params.jsonl")
    have_params = set()
    if os.path.exists(pfile):
        for line in open(pfile, encoding="utf-8"):
            try:
                have_params.add(json.loads(line)["slice"])
            except Exception:
                pass
    plog = open(pfile, "a", encoding="utf-8")
    for spath in slices:
        si = int(os.path.basename(spath)[6:10])
        out_img = os.path.join(out_dir, "%s_s%04d.jpg" % (aug_rec, si))
        out_iq = os.path.join(iq_dir, "slice_%04d.iq" % si)
        img_ok = os.path.exists(out_img)
        iq_ok = os.path.exists(out_iq)
        if img_ok and iq_ok:
            skipped += 1
            continue
        if shutil.disk_usage(r"E:\\").free < MIN_FREE_GB * 1e9:
            log("ABORT: E: free below %d GB" % MIN_FREE_GB)
            plog.close()
            sys.exit(2)
        try:
            if iq_ok:
                # IQ saved earlier, image missing -> just render
                y_n = os.path.getsize(out_iq) // 8
            else:
                raw = np.fromfile(spath, dtype=np.float32)
                n = raw.size // 2
                iq = np.empty(n, dtype=np.complex64)
                iq.real = raw[0::2]; iq.imag = raw[1::2]
                del raw
                rng = rng_for(tax, rec, si)
                y, params = augment(iq, fs, rng)
                del iq
                buf = np.empty(2 * len(y), dtype=np.float32)
                buf[0::2] = y.real; buf[1::2] = y.imag
                buf.tofile(out_iq)
                iq_bytes += buf.nbytes
                y_n = len(y)
                del y, buf
                if si not in have_params:
                    plog.write(json.dumps({"slice": si, "params": params}) + "\n")
            if not img_ok:
                duration = min(0.1, y_n / fs)
                imgs = generate_images(datapack=out_iq, fs=int(fs),
                                       stft_point=1024, duration_time=duration,
                                       location='buffer', file_type=np.float32)
                if not imgs:
                    failed += 1
                    log("  NO FRAME %s/%s s%04d" % (tax, rec, si))
                    continue
                imgs[0].convert("RGB").save(out_img, "JPEG", quality=95)
            done += 1
        except Exception as e:
            failed += 1
            log("  AUG FAIL %s/%s s%04d: %s" % (tax, rec, si, e))
        if (done + failed) % 100 == 0 and (done + failed) > 0:
            rate = done / max(time.time() - t0, 1)
            log("  progress: done=%d skipped=%d failed=%d iq_gb=%.1f (%.1f/min)"
                % (done, skipped, failed, iq_bytes / 1e9, rate * 60))
    plog.close()

log("AUGMENT DONE: rendered=%d skipped=%d failed=%d" % (done, skipped, failed))
if failed > 50:
    sys.exit(1)
