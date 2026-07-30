# A script to run inference on spectrogram images or raw frequency data.
import argparse

from utils.benchmark import ClassifyModel


def parse_args():
    parser = argparse.ArgumentParser(description='RFUAV inference entrypoint')
    parser.add_argument(
        '--task',
        choices=['classify', 'two_stage'],
        default='classify',
        help='inference task type',
    )
    parser.add_argument('--cfg', type=str, default='', help='classification config yaml path')
    parser.add_argument('--weight-path', type=str, default='', help='classification model weight path')
    parser.add_argument('--source', type=str, default='', help='input source path')
    parser.add_argument('--save-path', type=str, default='./res/', help='inference output directory')
    parser.add_argument('--two-stage-cfg', type=str, default='../example/two_stage/sample.json', help='two-stage json cfg path')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.task == 'classify':
        if not args.cfg or not args.weight_path:
            raise ValueError('`--cfg` and `--weight-path` are required for classify inference.')
        test = ClassifyModel(cfg=args.cfg, weight_path=args.weight_path)
        test.inference(source=args.source, save_path=args.save_path)
    else:
        # Imported lazily so the classify task doesn't pull in the detection
        # stack (DetModels -> vendored YOLOv5 -> requests/thop/torch).
        from utils.TwoStagesDetector import TwoStagesDetector
        TwoStagesDetector(cfg=args.two_stage_cfg)


if __name__ == '__main__':
    main()
