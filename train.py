# A script to train the detect model and classification model on spectrogram image.
from utils.trainer import CustomTrainer
from utils.trainer import DetTrainer


def main():

    # classification Trainer
    model = CustomTrainer(cfg='')
    model.train()

    # Detection Trainer
    save_dir = ''
    model = DetTrainer(model_name='yolo', dataset_dir = '')
    model.train(save_dir=save_dir)


if __name__ == '__main__':
    main()