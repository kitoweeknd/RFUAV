"""Chain 5: top up all three sources to 20 slices/recording -> render ->
manifest -> rebuild training tree -> inventory + summary.
Aborts on first non-zero exit; refuses to start under 170 GB free on E:."""
import os, shutil, subprocess, sys, time

PY = r"C:\ML\anaconda\envs\p0\python.exe"
TOOLS = r"D:\ML\RFUAV\tools"
LOG = os.path.join(TOOLS, "topup20_chain_log.txt")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def run(name):
    log("--- %s start ---" % name)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([PY, "-X", "utf8", "-u", os.path.join(TOOLS, name)],
                             stdout=f, stderr=f)
    log("--- %s exit=%d (%d s) ---" % (name, rc, time.time() - t0))
    if rc != 0:
        log("=== CHAIN ABORTED at %s ===" % name)
        sys.exit(rc)


free_gb = shutil.disk_usage(r"E:\\").free / 1e9
log("=== TOPUP-20 CHAIN START (E: free %.1f GB) ===" % free_gb)
if free_gb < 170:
    log("ABORT: need >= 170 GB free")
    sys.exit(2)

run("topup_rfuav.py")
run("topup_dronerfa.py")
run("stage2b_topup_dronerfb_20.py")
run("stage3_render_canonical.py")
run("stage4_master_manifest.py")
tds = r"E:\Drone_dataset\RFUAV-v2\train_dataset_v1"
if os.path.isdir(tds):
    shutil.rmtree(tds)
    log("removed train_dataset_v1 (hardlinks only) for clean rebuild")
run("stage5_build_train_split.py")
run("build_inventory.py")
run("gen_summary.py")
log("=== TOPUP-20 CHAIN COMPLETE ===")
