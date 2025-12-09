"""A script to evaluate the trained classify model on the benchmark dataset.

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

Returns:
    the evaluation metrics including accuracy, top-k accuracy, F1 score, and confusion matrix will be saved to the source path.
"""

from utils.benchmark import Classify_Model


def main():
    source = ''
    test = Classify_Model(cfg='',
                          weight_path='')
    # test.inference(source=source, save_path='./res/')  # for inference test
    test.benchmark(data_path=source)  # for benchmark test

    """ Two-stage detector test 
    cfg_path = '../example/two_stage/sample.json'
    TwoStagesDetector(cfg=cfg_path)
    """


if __name__ == '__main__':
    main()