"""Chain 8: rendering-scale ablation. scaled render -> scaled tree -> configs
-> train c8a (native control) -> train c8b (fixed-20MHz)."""
import os, shutil, subprocess, sys, time
PY = r"C:\ML\anaconda\envs\p0\python.exe"
TOOLS = r"D:\ML\RFUAV\tools"
LOG = os.path.join(TOOLS, "scale_ablation_log.txt")
def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))
def run(args, tag):
    log("--- %s start ---" % tag); t0 = time.time()
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([PY, "-X", "utf8", "-u"] + args, stdout=f, stderr=f)
    log("--- %s exit=%d (%d s) ---" % (tag, rc, time.time() - t0))
    if rc != 0:
        log("=== CHAIN 8 ABORTED at %s ===" % tag); sys.exit(rc)
free = shutil.disk_usage(r"E:\\").free / 1e9
log("=== SCALE ABLATION CHAIN START (E: free %.1f GB) ===" % free)
if free < 20:
    log("ABORT: low disk"); sys.exit(2)
run([os.path.join(TOOLS, "render_scaled.py")], "render_scaled")
run([os.path.join(TOOLS, "stage5b_build_scaled_split.py")], "stage5b_scaled_tree")
run([os.path.join(TOOLS, "make_c8_cfgs.py")], "make_c8_cfgs")
run([os.path.join(TOOLS, "train_runner.py"), r"D:\ML\RFUAV\configs\exp_c8a_native46.yaml",
     os.path.join(TOOLS, "train_c8a_log.txt")], "train_c8a_native")
run([os.path.join(TOOLS, "train_runner.py"), r"D:\ML\RFUAV\configs\exp_c8b_scaled46.yaml",
     os.path.join(TOOLS, "train_c8b_log.txt")], "train_c8b_scaled")
log("=== SCALE ABLATION CHAIN COMPLETE ===")
