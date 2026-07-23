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


def generator(pics_path, txt_path):
    pics = os.listdir(pics_path)
    for pic in pics:
        count = None
        for _ in name_table.keys():
            if _ in pic:
                count = name_table[_]
                break
        if count is None:
            print(f'Warning: no label key matched for {pic}, skipping.')
            continue
        full_path = os.path.join(txt_path, os.path.splitext(pic)[0] + '.txt')
        with open(full_path, 'w') as file:
            file.write(str(count) + ' ' + "0.5130208333333334 0.5045138888888889 0.775 0.7701388888888889")
        print(pic + ' labeled.')

    print("Label complete.")


def split_pics(source_dir, other_dir, train_dir, valid_dir, ratio):
    drone_type = [_ for _ in os.listdir(source_dir) if _ not in other_dir]
    for drone in drone_type:
        pics = [_ for _ in os.listdir(source_dir+drone + '/')]
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        random.shuffle(pics)
        split_point = int(len(pics) * ratio)
        valid_files = pics[:split_point]
        train_files = pics[split_point:]

        for file in valid_files:
            shutil.copy(os.path.join(source_dir+drone, file), os.path.join(valid_dir, file))
            print(file + ' copied to valid dir.')

        for file in train_files:
            shutil.copy(os.path.join(source_dir+drone, file), os.path.join(train_dir, file))
            print(file + ' copied to train dir.')


def main():

    source_dir = ''
    other_dir = []
    train_dir = ''
    valid_dir = ''
    print("splitting the TrainSet and ValidSet.")
    # split_pics(source_dir, other_dir, train_dir, valid_dir, 0.2)
    print("Done.")

    pics_valid_path = ""
    pics_train_path = ""
    pics_test_path = ""
    txt_valid_path = ''
    txt_train_path = ""
    txt_test_path = ""

    print('Generating the train labels.')
    generator(pics_train_path,  txt_train_path)
    print('Done.')
    print('Generating the valid labels.')
    generator(pics_valid_path, txt_valid_path)
    print('Done.')
    print('Generating the test labels.')
    generator(pics_test_path, txt_test_path)
    print('Done.')




if __name__ == '__main__':
    main()