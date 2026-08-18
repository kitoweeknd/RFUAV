"""Chain 6: channel-augment train slices (streamed, x1) -> build augmented
training tree + c5 config -> train c5. Lights-out overnight run."""
import os, subprocess, sys, time

PY = r"C:\ML\anaconda\envs\p0\python.exe"
TOOLS = r"D:\ML\RFUAV\tools"
LOG = os.path.join(TOOLS, "aug_chain_log.txt")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def run(args, tag):
    log("--- %s start ---" % tag)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([PY, "-X", "utf8", "-u"] + args, stdout=f, stderr=f)
    log("--- %s exit=%d (%d s) ---" % (tag, rc, time.time() - t0))
    if rc != 0:
        log("=== CHAIN ABORTED at %s ===" % tag)
        sys.exit(rc)


log("=== AUG CHAIN START ===")
run([os.path.join(TOOLS, "augment_render_train.py")], "augment_render_train")
run([os.path.join(TOOLS, "build_train_aug_tree.py")], "build_train_aug_tree")
run([os.path.join(TOOLS, "train_runner.py"),
     r"D:\ML\RFUAV\configs\exp_c5_resnet50_aug_v1.yaml",
     os.path.join(TOOLS, "train_c5_log.txt")], "train_c5")
log("=== AUG CHAIN COMPLETE ===")
