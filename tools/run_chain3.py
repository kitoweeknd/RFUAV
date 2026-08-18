"""Chain 3: quarantine twins -> topup DroneRFa -> render -> manifest v2 ->
rebuild train_dataset_v1. Aborts on first non-zero exit."""
import os, shutil, subprocess, sys, time

PY = r"C:\ML\anaconda\envs\p0\python.exe"
TOOLS = r"D:\ML\RFUAV\tools"
LOG = os.path.join(TOOLS, "topup_chain_log.txt")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def run(name):
    log("--- %s start ---" % name)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([PY, "-X", "utf8", os.path.join(TOOLS, name)],
                             stdout=f, stderr=f)
    log("--- %s exit=%d (%d s) ---" % (name, rc, time.time() - t0))
    if rc != 0:
        log("=== CHAIN ABORTED at %s ===" % name)
        sys.exit(rc)


log("=== TOPUP CHAIN START ===")
run("quarantine_rfa_twins.py")
run("topup_dronerfa.py")
run("stage3_render_canonical.py")
run("stage4_master_manifest.py")
tds = r"E:\Drone_dataset\RFUAV-v2\train_dataset_v1"
if os.path.isdir(tds):
    shutil.rmtree(tds)
    log("removed train_dataset_v1 (hardlinks only) for clean rebuild")
run("stage5_build_train_split.py")
log("=== TOPUP CHAIN COMPLETE ===")
