# A train sample code
from utils.trainer import CustomTrainer
# from utils.trainer import DetTrainer


def main():

    model = CustomTrainer(cfg='./configs/exp2.1_ResNet101.yaml')
    model.train()

    # train a custom signal detect model
    # model = DetTrainer(model_name='yolo')
    # model.train(save_dir='E:/Drone_dataset/RFUAV/signal_detect/test/')


if __name__ == '__main__':
    main()