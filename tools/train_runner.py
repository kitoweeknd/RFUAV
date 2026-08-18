"""Generic detached training runner: train_runner.py <config.yaml> <logfile>"""
import sys, traceback

sys.path.insert(0, r"D:\ML\RFUAV")


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    def write(self, s):
        self.f.write(s)

    def flush(self):
        self.f.flush()


if __name__ == "__main__":
    cfg, log = sys.argv[1], sys.argv[2]
    sys.stdout = sys.stderr = Tee(log)
    print("=== train start: %s ===" % cfg)
    try:
        from utils.trainer import CustomTrainer
        trainer = CustomTrainer(cfg=cfg)
        trainer.train()
        print("=== train DONE ===")
    except Exception:
        traceback.print_exc()
        print("=== train FAILED ===")
