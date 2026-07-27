# A script to train classification models on spectrogram images.
import argparse

from utils.trainer import CustomTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='RFUAV classification training entrypoint')
    parser.add_argument('--cfg', type=str, required=True, help='classification config yaml path')
    return parser.parse_args()


def main():
    args = parse_args()
    model = CustomTrainer(cfg=args.cfg)
    model.train()


if __name__ == '__main__':
    main()
