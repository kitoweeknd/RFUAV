"""Smoke-train runner for the canonical v1 grouped split (see configs/train_canonical_v1.yaml)."""
import sys, traceback

sys.path.insert(0, r"D:\ML\RFUAV")
LOG = r"D:\ML\RFUAV\tools\train_canonical_v1_log.txt"


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    def write(self, s):
        self.f.write(s)

    def flush(self):
        self.f.flush()


if __name__ == "__main__":
    sys.stdout = sys.stderr = Tee(LOG)
    print("=== smoke train start ===")
    try:
        from utils.trainer import CustomTrainer
        trainer = CustomTrainer(cfg=r"D:\ML\RFUAV\configs\train_canonical_v1.yaml")
        trainer.train()
        print("=== smoke train DONE ===")
    except Exception:
        traceback.print_exc()
        print("=== smoke train FAILED ===")
