"""Augmentations for image data augmentation
All the augmentations methods are provided by albumentations: https://github.com/albumentations-team/albumentations

Args:
    dataset_path (str): The file path of the dataset.
    methods (list[str], optional): The augmentation methods to apply. Default methods include:
        1. AdvancedBlur
        2. CLAHE
        3. ColorJitter
        4. GaussNoise
        5. ISONoise
        6. Sharpen
    output_path (str, optional): The path where the augmented dataset will be saved. If not specified, a new directory named `dataset_aug` will be created in the same directory as the dataset.
"""

import albumentations as A
import shutil
import random
import cv2
from PIL import Image
import numpy as np
import os


def data_augmentation(dataset_path: str = None,
                      output_path: str = None,
                      methods: list[str] = None):
    """
    Perform data augmentation on the given dataset using specified methods.

    Args:
        dataset_path (str): The file path of the dataset.
        output_path (str, optional): The path where the augmented dataset will be saved. If not specified, a new directory named `dataset_aug` will be created in the same directory as the dataset.
        methods (list, optional): The augmentation methods to apply. Each method is an instance of an Albumentations transformation. If not specified, default methods are used.

    Raises:
        FileNotFoundError: If the dataset path does not exist.

    Your dataset should organize like this
    dataset_path
        ├──train
        │└──images
        └──valid
         └──images
    """

    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset path does not exist")

    if not output_path:
        prefix = os.path.dirname(os.path.dirname(dataset_path))
        output_path = os.path.join(prefix, 'dataset_aug')
        os.mkdir(output_path)

    if methods is None:
        methods = [
            A.AdvancedBlur(
                blur_limit=(7, 13),
                sigma_x_limit=(7, 13),
                sigma_y_limit=(7, 13),
                rotate_limit=(-90, 90),
                beta_limit=(0.5, 8),
                noise_limit=(2, 10),
                p=1),
            A.CLAHE(
                clip_limit=3,
                tile_grid_size=(13, 13),
                p=1),
            A.ColorJitter(
                brightness=(0.5, 1.5),
                contrast=(1, 1),
                saturation=(1, 1),
                hue=(-0, 0),
                p=1),
            A.GaussNoise(
                var_limit=(100, 500),
                mean=0,
                p=1),
            A.ISONoise(
                intensity=(0.2, 0.5),
                color_shift=(0.01, 0.05),
                p=1),
            A.Sharpen(
                alpha=(0.2, 0.5),
                lightness=(0.5, 1),
                p=1)
            ]

    total_path = [
        os.path.join(dataset_path, 'train'),
        os.path.join(dataset_path, 'valid'),
    ]

    for path in total_path:
        _ = os.path.join(output_path, path.split('/')[-1])
        if not os.path.exists(_):
            os.mkdir(_)
        classes = os.listdir(path)

        for _class in classes:
            _save_path = os.path.join(_, _class)
            if not os.path.exists(_save_path):
                os.mkdir(_save_path)
            path_image = os.path.join(path, _class)
            images = os.listdir(path_image)
            i = 0

            for method in methods:

                for image in images:
                    original_image = cv2.imread(os.path.join(path_image, image))
                    # original_image = cv2.cvtColor(cv2.imread(os.path.join(path_image, image)), cv2.COLOR_BGR2RGB)
                    # show_image(original_image)
                    transform = A.Compose(method)
                    augmented = transform(image=original_image)
                    cv2.imwrite(os.path.join(_save_path, image+'_AugM'+str(i)+os.path.splitext(image)[1]), augmented['image'])
                    cv2.imwrite(os.path.join(_save_path, image+'_origin'+os.path.splitext(image)[1]), original_image)
                i += 1

            print(f"Finished augmentation of {_class}")


def show_image(image):
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 600)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def split_images(input_path, output_path, train_ratio=0.8):

    print('starting split')

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    for drone_type in os.listdir(input_path):

        train_path = os.path.join(os.path.join(output_path, 'train'), drone_type)
        valid_path = os.path.join(os.path.join(output_path, 'valid'), drone_type)

        if not os.path.exists(train_path):
            os.makedirs(train_path)

        if not os.path.exists(valid_path):
            os.makedirs(valid_path)

        image_files = [f for f in os.listdir(os.path.join(input_path, drone_type)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_files)

        num_train = int(len(image_files) * train_ratio)

        for image in image_files[:num_train]:
            src = os.path.join(os.path.join(input_path, drone_type), image)
            dst = os.path.join(train_path, image)
            shutil.copy(src, dst)

        for image in image_files[num_train:]:
            src = os.path.join(os.path.join(input_path, drone_type), image)
            dst = os.path.join(valid_path, image)
            shutil.copy(src, dst)

        print(drone_type + ' Done')



def read_image_with_chinese_path(image_path):

    pil_image = Image.open(image_path)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return cv_image


def crop_and_save_image(image,
                        output_path,
                        x,
                        y,
                        width,
                        height):
    """
        Crops a rectangular window from a fixed position in the image and saves the cropped content to the specified location.

        Args:
            image (str): Path to the input image.
            output_path (str): Path to save the output image.
            x (int): x-coordinate of the top-left corner of the rectangular window.
            y (int): y-coordinate of the top-left corner of the rectangular window.
            width (int): Width of the rectangular window.
            height (int): Height of the rectangular window.

        Returns:
            None
    """
    # 截取矩形窗
    cropped_image = image[y:y+height, x:x+width]

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存截取的图像
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved to {output_path}")


def CropImage(
        fig_save_path: str,
        file_path: str,
        x,
        y,
        width,
        height
):

    drones = os.listdir(file_path)

    for drone in drones:
        packlist = os.listdir(file_path + drone)
        for pack in packlist:

            _ = os.path.join(file_path+drone, pack)

            if os.path.isfile(_):
                check_folder(fig_save_path + drone)
                crop_and_save_image(read_image_with_chinese_path(_), fig_save_path + drone + '/' + pack, x=x, y=y, width=width, height=height)

            else:
                imgs = os.listdir(_)
                for img in imgs:
                    _save_path = os.path.join(fig_save_path + drone, pack)
                    check_folder(_save_path)
                    img_path = os.path.join(_, img)
                    crop_and_save_image(read_image_with_chinese_path(img_path), _save_path + '/' + img, x=x, y=y, width=width, height=height)
                    print(img + ' Done')
            print(pack + ' Done')
        print(drone + ' Done')
    print('All Done')


def check_folder(folder_path):
    """
    Checks and creates the folder if it does not exist.

    Parameters:
    - folder_path (str): Path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"folder '{folder_path}' created。")
    else:
        print(f"folder '{folder_path}' existed。")


# Usage---------------------------------------------------------------------------------------------------------------
def main():

    """
    split_images(input_path='', output_path='')
    """

    data_path = ""
    output_path = ""
    methods = [A.AdvancedBlur(
                    blur_limit=(7, 13),
                    sigma_x_limit=(7, 13),
                    sigma_y_limit=(7, 13),
                    rotate_limit=(-90, 90),
                    beta_limit=(0.5, 8),
                    noise_limit=(2, 10),
                    p=1),
               A.CLAHE(
                   clip_limit=3,
                   tile_grid_size=(13, 13),
                   p=1),
               A.ColorJitter(
                   brightness=(0.5, 1.5),
                   contrast=(1, 1),
                   saturation=(1, 1),
                   hue=(-0, 0),
                   p=1,
               ),
               A.GaussNoise(
                   var_limit=(100, 500),
                   mean=0,
                   p=1
               ),
               A.ISONoise(
                   intensity=(0.2, 0.5),
                   color_shift=(0.01, 0.05),
                   p=1
               ),
               A.Sharpen(
                   alpha=(0.2, 0.5),
                   lightness=(0.5, 1),
                   p=1
               )
               ]
    data_augmentation(dataset_path=data_path, output_path=output_path, methods=methods)


# test
if __name__ == '__main__':
    main()