"""Chain 10: benchmark array, one model per family. Sequential; a failed
model is logged and skipped (no chain abort). Per-epoch checkpoints trimmed
after each model to protect disk."""
import glob, os, subprocess, sys, time
PY = r"C:\ML\anaconda\envs\p0\python.exe"
TOOLS = r"D:\ML\RFUAV\tools"
RUNS = r"E:\Drone_dataset\RFUAV-v2\benchmark\runs"
LOG = os.path.join(TOOLS, "bench_chain_log.txt")
MODELS = ["mobilenetv3_large_100", "vit_base_patch16_224",
          "swinv2_tiny_window8_256", "convnext_tiny",
          "efficientnetv2_rw_s", "maxvit_tiny_tf_224",
          "eva02_small_patch14_224"]
def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))
log("=== BENCHMARK CHAIN START (%d models) ===" % len(MODELS))
rc0 = subprocess.call([PY, "-X", "utf8", os.path.join(TOOLS, "make_bench_cfgs.py")])
if rc0 != 0:
    log("ABORT: cfg generation failed"); sys.exit(1)
results = []
for m in MODELS:
    cfg = r"D:\ML\RFUAV\configs\bm_" + m + ".yaml"
    tlog = os.path.join(TOOLS, "train_bm_%s.txt" % m)
    log("--- %s start ---" % m); t0 = time.time()
    with open(os.path.join(TOOLS, "bench_chain_log.txt"), "a") as devnull:
        rc = subprocess.call([PY, "-X", "utf8", "-u",
                              os.path.join(TOOLS, "train_runner.py"), cfg, tlog],
                             stdout=devnull, stderr=devnull)
    dt = (time.time() - t0) / 3600
    # trim per-epoch checkpoints, keep best_model.pth
    n_rm = 0
    for f in glob.glob(os.path.join(RUNS, "bm_" + m, "*_epoch_*.pth")):
        os.remove(f); n_rm += 1
    status = "ok" if rc == 0 else "FAILED(rc=%d)" % rc
    results.append((m, status, dt))
    log("--- %s %s (%.2f h, trimmed %d ckpts) ---" % (m, status, dt, n_rm))
log("=== BENCHMARK CHAIN COMPLETE: " +
    ", ".join("%s:%s" % (m, s) for m, s, _ in results) + " ===")
