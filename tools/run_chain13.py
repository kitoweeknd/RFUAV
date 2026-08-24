"""Chain 13: TODO sequence in order — (1) 3-seed replicates, (2) SwinV2 x
augmentation, (3) individual-ID benchmark, (4) cross-source tier. All t256,
batch 64. Failures logged, chain continues."""
import glob, os, subprocess, sys, time
PY = r"C:\ML\anaconda\envs\p0\python.exe"
TOOLS = r"D:\ML\RFUAV\tools"
RUNS = r"E:\Drone_dataset\RFUAV-v2\benchmark\runs"
LOG = os.path.join(TOOLS, "seq_chain_log.txt")
def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))
def run(args, tag):
    log("--- %s start ---" % tag); t0 = time.time()
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([PY, "-X", "utf8", "-u"] + args, stdout=f, stderr=f)
    log("--- %s exit=%d (%.2f h) ---" % (tag, rc, (time.time()-t0)/3600))
    return rc
log("=== SEQ CHAIN START ===")
if run([os.path.join(TOOLS, "build_seq_trees.py")], "build_seq_trees") != 0:
    log("ABORT: trees"); sys.exit(1)
if run([os.path.join(TOOLS, "make_seq_cfgs.py")], "make_seq_cfgs") != 0:
    log("ABORT: cfgs"); sys.exit(1)
JOBS = (["sd_%s_s%d" % (m, s) for m in
         ("swinv2_tiny_window8_256", "vit_base_patch16_224", "mobilenetv3_large_100")
         for s in (1, 2, 3)]
        + ["ag_swinv2_s1", "ag_swinv2_s2", "id18_swinv2", "id18_resnet50",
           "id6_swinv2", "cs_swinv2"])
for j in JOBS:
    rc = run([os.path.join(TOOLS, "train_runner.py"),
              r"D:\ML\RFUAV\configs" + "\\" + j + ".yaml",
              os.path.join(TOOLS, "train_%s.txt" % j)], "train_" + j)
    for f in glob.glob(os.path.join(RUNS, j, "*_epoch_*.pth")):
        os.remove(f)
    log("%s rc=%d (ckpts trimmed)" % (j, rc))
for j in ("id18_swinv2", "cs_swinv2"):
    run([os.path.join(TOOLS, "eval_percls.py"),
         r"D:\ML\RFUAV\configs" + "\\" + j + ".yaml"], "eval_" + j)
log("=== SEQ CHAIN COMPLETE ===")
