# A script to train classification/detection models on spectrogram images.
import argparse

from utils.trainer import CustomTrainer, DetTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='RFUAV training entrypoint')
    parser.add_argument(
        '--task',
        choices=['classify', 'detect', 'both'],
        default='classify',
        help='training task type',
    )
    parser.add_argument('--cfg', type=str, default='', help='classification config yaml path')
    parser.add_argument('--det-model', type=str, default='yolo', help='detection model name')
    parser.add_argument('--dataset-dir', type=str, default='', help='detection dataset dir')
    parser.add_argument('--det-save-dir', type=str, default='', help='detection save dir')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task in ('classify', 'both'):
        if not args.cfg:
            raise ValueError('`--cfg` is required for classification training.')
        model = CustomTrainer(cfg=args.cfg)
        model.train()

    if args.task in ('detect', 'both'):
        model = DetTrainer(model_name=args.det_model, dataset_dir=args.dataset_dir)
        model.train(save_dir=args.det_save_dir)


if __name__ == '__main__':
    main()