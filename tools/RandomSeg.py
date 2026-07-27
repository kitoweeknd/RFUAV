"""random crop the dataset into train set and valid set
"""
import os
import random
import shutil


def split_images(source_dir, target_dir, other_dir, open_dir, close_dir, ratio, special_dir, seed=42):

    # Ratio convention: `split_point = int(len(pics) * ratio)` and the first
    # `ratio` fraction of the shuffled files goes to the first split
    # (train / open_dir); the remaining `1 - ratio` goes to valid / close_dir.
    # NOTE (correlation leakage): because splitting happens at the frame level,
    # consecutive frames from a single continuous recording can end up in both
    # the train and valid splits, which leaks correlated content across them.
    # Seed the RNG so the split is reproducible across runs (files are copied,
    # not moved, so an unseeded reshuffle would accumulate duplicates / leakage).
    random.seed(seed)

    drone_type = [_ for _ in os.listdir(source_dir) if _ not in other_dir]

    for drone in drone_type:
        i = 0
        if drone in special_dir:
            base = os.path.join(source_dir, drone, target_dir)
            for detail_type in os.listdir(base):
                src = os.path.join(base, detail_type)
                pics = os.listdir(src)
                os.makedirs(os.path.join(open_dir, detail_type), exist_ok=True)
                os.makedirs(os.path.join(close_dir, detail_type), exist_ok=True)
                random.shuffle(pics)
                split_point = int(len(pics) * ratio)
                open_files = pics[:split_point]
                close_files = pics[split_point:]

                for file in open_files:
                    shutil.copy(os.path.join(src, file), os.path.join(open_dir, detail_type, file))
                for file in close_files:
                    shutil.copy(os.path.join(src, file), os.path.join(close_dir, detail_type, file))

        else:
            train_dir = os.path.join(target_dir, 'train', drone)
            valid_dir = os.path.join(target_dir, 'valid', drone)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(valid_dir, exist_ok=True)
            packs = os.listdir(os.path.join(source_dir, drone))
            for pack in packs:
                pics = os.listdir(os.path.join(source_dir, drone, pack))
                random.shuffle(pics)
                split_point = int(len(pics) * ratio)
                train = pics[:split_point]
                val = pics[split_point:]

                # Copy straight to the renamed target (preserve the source
                # extension instead of forcing .jpg; one op instead of copy+rename).
                for file in train:
                    ext = os.path.splitext(file)[1]
                    shutil.copy(os.path.join(source_dir, drone, pack, file),
                                os.path.join(train_dir, f'{drone}{i}{ext}'))
                    i += 1
                for file in val:
                    ext = os.path.splitext(file)[1]
                    shutil.copy(os.path.join(source_dir, drone, pack, file),
                                os.path.join(valid_dir, f'{drone}{i}{ext}'))
                    i += 1
            print(f'{packs} done')
        print(f'{drone} done')


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Randomly split a per-drone image dataset into train/valid (copying files).")
    parser.add_argument("source_dir", help="Root dir containing one subfolder per drone type.")
    parser.add_argument("target_dir", help="Output dir (train/ and valid/ are created under it).")
    parser.add_argument("--ratio", type=float, default=0.3,
                        help="Fraction of each folder that goes to the first split (default 0.3).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible splits.")
    parser.add_argument("--other-dir", nargs="*", default=[],
                        help="Subfolder names under source_dir to skip.")
    parser.add_argument("--special-dir", nargs="*", default=[],
                        help="Drone folders handled via the open/close split instead of train/valid.")
    parser.add_argument("--open-dir", default="", help="Output dir for the 'open' split (special dirs).")
    parser.add_argument("--close-dir", default="", help="Output dir for the 'close' split (special dirs).")
    args = parser.parse_args()

    split_images(args.source_dir, args.target_dir, args.other_dir, args.open_dir,
                 args.close_dir, args.ratio, args.special_dir, seed=args.seed)


if __name__ == '__main__':
    main()