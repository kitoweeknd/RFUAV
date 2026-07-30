"""generate the label for yolo
"""
import os
import random
import shutil


# label table
name_table = {
    'DAUTEL': 0,
    'DEVENTION': 1,
    'DJFPVCOMBO': 2,
    'DJI AVATA': 3,
    'DJMAVIC3PRO': 4,
    'DJMINI3': 5,
    'DJMINI4PRO': 6,
    'FLYSKY_EL_18': 7,
    'FLYSKY_16X': 8,
    'FLYSKY_NV14': 9,
    'FRSKY-X9DP20': 10,
    'FRSKY-X14': 11,
    'FRSKY-X20R': 12,
    'FUTABA-T10J': 13,
    'FutabaT14SG': 14,
    'FUTABA-T16IZ': 15,
    'FUTABA-T18SZ': 16,
    'herelink': 17,
    'JR PROPO_XG7': 18,
    'JR PROPO_XG14': 19,
    'JUMPER-T14': 20,
    'JUMPER-TProV2': 21,
    'Radiamaster-BOXER': 22,
    'Radiamaster_TX16S': 23,
    'Radiolink_AT9S_pro': 24,
    'Radiolink_AT10_II': 25,
    'SIYI_FT24': 26,
    'SIYI_MK15': 27,
    'SIYI_MK32': 28,
    'SKYDROID-H12': 29,
    'SKYDROID-T10': 30,
}


# YOLO box (normalized cx cy w h). The signal ROI is fixed for this spectrogram
# dataset, so every image gets the same box; edit here if boxes should vary.
DEFAULT_BBOX = "0.5130208333333334 0.5045138888888889 0.775 0.7701388888888889"


def generator(pics_path, txt_path):
    os.makedirs(txt_path, exist_ok=True)
    for pic in os.listdir(pics_path):
        count = None
        for key in name_table:
            if key in pic:
                count = name_table[key]
                break
        if count is None:
            print(f'Warning: no label key matched for {pic}, skipping.')
            continue
        full_path = os.path.join(txt_path, os.path.splitext(pic)[0] + '.txt')
        with open(full_path, 'w') as file:
            file.write(f"{count} {DEFAULT_BBOX}")
        print(pic + ' labeled.')

    print("Label complete.")


def split_pics(source_dir, other_dir, train_dir, valid_dir, ratio, seed=42):
    random.seed(seed)  # reproducible split
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    # Only descend into drone-type subfolders; a stray file in source_dir would
    # otherwise crash the os.listdir(join(source_dir, drone)) call below.
    drone_type = [d for d in os.listdir(source_dir)
                  if d not in other_dir and os.path.isdir(os.path.join(source_dir, d))]
    for drone in drone_type:
        pics = os.listdir(os.path.join(source_dir, drone))
        random.shuffle(pics)
        split_point = int(len(pics) * ratio)
        valid_files = pics[:split_point]
        train_files = pics[split_point:]

        # Prefix with the drone folder so identical frame filenames across drones
        # don't silently overwrite each other in the flat train/valid dirs.
        for file in valid_files:
            shutil.copy(os.path.join(source_dir, drone, file),
                        os.path.join(valid_dir, f'{drone}_{file}'))
        for file in train_files:
            shutil.copy(os.path.join(source_dir, drone, file),
                        os.path.join(train_dir, f'{drone}_{file}'))
        print(f'{drone} done')


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate YOLO labels from image filenames, or split a dataset.")
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_label = sub.add_parser('label', help="Write a YOLO .txt label per image in a folder.")
    p_label.add_argument('pics_path', help="Folder of images to label.")
    p_label.add_argument('txt_path', help="Output folder for the .txt labels.")

    p_split = sub.add_parser('split', help="Split a per-drone image dir into train/valid.")
    p_split.add_argument('source_dir', help="Root dir with one subfolder per drone.")
    p_split.add_argument('train_dir')
    p_split.add_argument('valid_dir')
    p_split.add_argument('--ratio', type=float, default=0.2,
                         help="Fraction going to valid (default 0.2).")
    p_split.add_argument('--seed', type=int, default=42, help="RNG seed.")
    p_split.add_argument('--other-dir', nargs='*', default=[],
                         help="Subfolder names to skip.")

    args = parser.parse_args()
    if args.cmd == 'label':
        generator(args.pics_path, args.txt_path)
    else:
        split_pics(args.source_dir, args.other_dir, args.train_dir,
                   args.valid_dir, args.ratio, seed=args.seed)


if __name__ == '__main__':
    main()