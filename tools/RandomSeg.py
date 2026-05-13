"""random crop the dataset into train set and valid set
"""
import os
import random
import shutil


def split_images(source_dir, target_dir, other_dir, open_dir, close_dir, ratio, special_dir):

    drone_type = [_ for _ in os.listdir(source_dir) if _ not in other_dir]

    for drone in drone_type:
        i = 0
        if drone in special_dir:
            for detail_type in os.listdir(source_dir+drone+'/'+target_dir+'/'):
                pics = [_ for _ in os.listdir(source_dir+drone+'/'+target_dir+'/'+detail_type+'/')]
                os.makedirs(open_dir+detail_type, exist_ok=True)
                os.makedirs(close_dir+detail_type, exist_ok=True)
                random.shuffle(pics)
                split_point = int(len(pics) * ratio)
                open_files = pics[:split_point]
                close_files = pics[split_point:]

                for file in open_files:
                    shutil.copy(os.path.join(source_dir+drone+'/'+target_dir+'/'+detail_type, file), os.path.join(open_dir+detail_type, file))

                for file in close_files:
                    shutil.copy(os.path.join(source_dir+drone+'/'+target_dir+'/'+detail_type, file), os.path.join(close_dir+detail_type, file))

        else:
            packs = os.listdir(os.path.join(source_dir, drone))
            for pack in packs:
                pics = [_ for _ in os.listdir(os.path.join(source_dir, drone, pack))]
                if not os.path.exists(os.path.join(target_dir, 'train', drone)):
                    check_path(os.path.join(target_dir, 'train', drone))
                if not os.path.exists(os.path.join(target_dir, 'valid', drone)):
                    check_path(os.path.join(target_dir, 'valid', drone))
                random.shuffle(pics)
                split_point = int(len(pics) * ratio)
                train = pics[:split_point]
                val = pics[split_point:]

                for file in train:
                    shutil.copy(os.path.join(source_dir, drone, pack, file), os.path.join(target_dir, 'train', drone))
                    os.rename(os.path.join(target_dir, 'train', drone, file), os.path.join(target_dir, 'train', drone, drone + str(i) + '.jpg'))
                    i += 1
                for file in val:
                    shutil.copy(os.path.join(source_dir, drone, pack, file), os.path.join(target_dir, 'valid', drone))
                    os.rename(os.path.join(target_dir, 'valid', drone, file), os.path.join(target_dir, 'valid', drone, drone + str(i) + '.jpg'))
                    i += 1
            print(f'{packs} done')
        print(f'{drone} done')


def check_path(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"dir created: {directory_path}")
    else:
        print(f"dir exits: {directory_path}")


def main():
    source_directory = 'Your data path'
    target_dir = 'Your target dir'
    other_dir = []
    ratio = 0.3
    open_dir = ''
    close_dir = ''

    special_dir = []

    split_images(source_directory, target_dir, other_dir, open_dir, close_dir, ratio, special_dir)


if __name__ == '__main__':
    main()