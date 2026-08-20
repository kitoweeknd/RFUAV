"""Chain 9: c7 augmentation retest at healthy micro-batch (4+accum16, matches
the fresh c8a native-46 control). Tops up augmentation for the 12 new train
recordings, rebuilds the 46-class augmented tree, trains c7."""
import os, shutil, subprocess, sys, time
PY = r"C:\ML\anaconda\envs\p0\python.exe"
TOOLS = r"D:\ML\RFUAV\tools"
LOG = os.path.join(TOOLS, "c7_chain_log.txt")
def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))
def run(args, tag):
    log("--- %s start ---" % tag); t0 = time.time()
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([PY, "-X", "utf8", "-u"] + args, stdout=f, stderr=f)
    log("--- %s exit=%d (%d s) ---" % (tag, rc, time.time() - t0))
    if rc != 0:
        log("=== CHAIN 9 ABORTED at %s ===" % tag); sys.exit(rc)
log("=== C7 CHAIN START ===")
run([os.path.join(TOOLS, "augment_render_train.py")], "augment_topup_new_classes")
stale = r"E:\Drone_dataset\RFUAV-v2\benchmark\train_dataset_v1_aug"
if os.path.isdir(stale):
    shutil.rmtree(stale)
    log("removed stale 44-class aug tree (hardlinks only)")
run([os.path.join(TOOLS, "build_train_aug_tree.py")], "build_aug_tree_46")
run([os.path.join(TOOLS, "train_runner.py"), r"D:\ML\RFUAV\configs\exp_c7_resnet50_aug46.yaml",
     os.path.join(TOOLS, "train_c7_log.txt")], "train_c7")
log("=== C7 CHAIN COMPLETE ===")
