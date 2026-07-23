from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch

from utils.metrics import plt_settings


class ConfusionMatrix:
    """
    A class for calculating a confusion matrix. origin: https://github.com/WongKinYiu/yolov9

    Attributes:
        matrix (np.array): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        pic_name: confusion matrix picture name
    """

    def __init__(self, nc, pic_name):
        self.pic_name = pic_name
        self.matrix = np.zeros((nc, nc))
        self.nc = nc

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task.

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """

        _preds = []
        for _ in preds:
            _preds.append(_.argmax())
        _preds = torch.tensor(_preds)

        for p, t in zip(_preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    @plt_settings({"font.size": 12})
    def plot(self, normalize=True, save_dir='', names=()):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (list(names)) if labels else 'auto'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Reds',
                       fmt='.2f' if normalize else '.0f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels,
                       cbar=True).set_facecolor((1, 1, 1))
        title = 'Confusion Matrix' + ' Normalized' * normalize
        ax.set_xlabel('GT')
        ax.set_ylabel('Pred')
        ax.set_title(title)
        plot_fname = Path(save_dir) / f'{self.pic_name}_{title.lower().replace(" ", "_")}.png'
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)


# Usage-----------------------------------------------------------------------------------------------------------------
def main():
    # probability matrix for each pred image
    pred = torch.tensor([[0, 0.9, 0.8, 0.3, 0.6],
                         [0, 0.9, 0.8, 0.7, 0.6],
                         [0, 0.9, 0.8, 0.7, 0.5],
                         [0, 0.9, 0.8, 0.1, 0.6],
                         [0, 0.9, 0.8, 0.7, 0.4]])
    targets = torch.tensor([1, 3, 4, 1, 2])  # labels for each image

    save_dir = ''

    test = ConfusionMatrix(nc=5, pic_name='demo')
    test.process_cls_preds(pred, targets)
    names = ('A', 'B', 'C', 'D', 'E')
    for _ in True, False:
        test.plot(save_dir=save_dir,
                  names=names,
                  normalize=_,)


if __name__ == '__main__':
    main()
