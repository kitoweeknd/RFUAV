"""Patch chain: missing uncut import + orphan quarantine -> render -> manifest."""
import subprocess, sys, time, os
PY = r"C:\ML\anaconda\envs\p0\python.exe"
TOOLS = r"D:\ML\RFUAV\tools"
LOG = os.path.join(TOOLS, "integration_chain2_log.txt")
STAGES = ["stage1c_import_missing.py", "stage3_render_canonical.py", "stage4_master_manifest.py"]
def log(msg):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
log("=== CHAIN2 START ===")
for stage in STAGES:
    log("--- %s start ---" % stage)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as lf:
        rc = subprocess.call([PY, os.path.join(TOOLS, stage)], stdout=lf, stderr=subprocess.STDOUT)
    log("--- %s exit=%d (%.0f s) ---" % (stage, rc, time.time() - t0))
    if rc != 0:
        log("=== CHAIN2 ABORTED at %s ===" % stage)
        sys.exit(rc)
log("=== CHAIN2 COMPLETE ===")
