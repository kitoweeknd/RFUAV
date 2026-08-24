"""Chain 11: naive-vs-grouped headline experiment. Build slice-level random
split, then train ResNet50 + SwinV2-T on it (batch 64)."""
import glob, os, subprocess, sys, time
PY = r"C:\ML\anaconda\envs\p0\python.exe"
TOOLS = r"D:\ML\RFUAV\tools"
RUNS = r"E:\Drone_dataset\RFUAV-v2\benchmark\runs"
LOG = os.path.join(TOOLS, "naive_chain_log.txt")
def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))
def run(args, tag):
    log("--- %s start ---" % tag); t0 = time.time()
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([PY, "-X", "utf8", "-u"] + args, stdout=f, stderr=f)
    log("--- %s exit=%d (%.2f h) ---" % (tag, rc, (time.time() - t0) / 3600))
    return rc
log("=== NAIVE-VS-GROUPED CHAIN START ===")
if run([os.path.join(TOOLS, "build_naive_split.py")], "build_naive_split") != 0:
    log("ABORT"); sys.exit(1)
if run([os.path.join(TOOLS, "make_naive_cfgs.py")], "make_naive_cfgs") != 0:
    log("ABORT"); sys.exit(1)
for m in ("resnet50", "swinv2_tiny_window8_256"):
    rc = run([os.path.join(TOOLS, "train_runner.py"),
              r"D:\ML\RFUAV\configs\nv_" + m + ".yaml",
              os.path.join(TOOLS, "train_nv_%s.txt" % m)], "train_nv_" + m)
    n_rm = 0
    for f in glob.glob(os.path.join(RUNS, "nv_" + m, "*_epoch_*.pth")):
        os.remove(f); n_rm += 1
    log("%s done rc=%d (trimmed %d ckpts)" % (m, rc, n_rm))
log("=== NAIVE-VS-GROUPED CHAIN COMPLETE ===")
