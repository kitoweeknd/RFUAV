"""A script to evaluate the trained classify model on benchmark dataset.
Args:
    source (str): The path to the benchmark dataset. Notice that the dataset should be organized like this:
        source
        ├──snr1
        │   ├──CM1
        │   │   ├──img1.jpg
        │   │   ├──img2.jpg
        │   │   └──...
        │   ├──CM2
        │   │   ├──img1.jpg
        │   │   ├──img2.jpg
        │   │   └──...
        │   └──...
        ├──snr2
        │   ├──CM1
        │   │   ├──img1.jpg
        │   └──...
        └──...
    cfg (str): The path to the configuration file.
    weight_path (str): The path to the weights file.
"""
import argparse



from utils.benchmark import ClassifyModel

def parse_args():
    parser = argparse.ArgumentParser(description='RFUAV benchmark entrypoint')
    parser.add_argument('--cfg', type=str, default='', help='classification config yaml path')
    parser.add_argument('--weight-path', type=str, default='', help='classification model weight path')
    parser.add_argument('--data-path', type=str, default='', help='benchmark dataset path')
    parser.add_argument('--save-path', type=str, default=None, help='benchmark result output path')
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.cfg or not args.weight_path or not args.data_path:
        raise ValueError('`--cfg`, `--weight-path`, and `--data-path` are required for benchmark.')

    test = ClassifyModel(cfg=args.cfg, weight_path=args.weight_path)
    test.benchmark(data_path=args.data_path, save_path=args.save_path)


if __name__ == '__main__':
    main()
