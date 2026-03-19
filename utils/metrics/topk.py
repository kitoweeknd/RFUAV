"""
origin: https://github.com/open-mmlab/mmeval
"""
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):
    """Top-k accuracy evaluation metric.

    This metric computes the accuracy based on the given topk and thresholds.

    Currently, this metric supports 5 kinds of inputs, i.e. ``numpy.ndarray``,
    ``torch.Tensor``, ``oneflow.Tensor``, ``tensorflow.Tensor`` and
    ``paddle.Tensor``, and the implementation for the calculation depends on
    the inputs type.

    Args:
        topk (int | Sequence[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Sequence[float | None] | float | None): Predictions with scores
            under the thresholds are considered negative. None means no
            thresholds. Defaults to 0.
        **kwargs: Keyword parameters passed to :class:`BaseMetric`.
    """

    def __init__(self,
                 topk: Union[int, Sequence[int]] = (1, ),
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if isinstance(topk, int):
            self.topk = (topk, )
        else:
            self.topk = tuple(topk)  # type: ignore
        self.maxk = max(self.topk)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs, )
        else:
            self.thrs = tuple(thrs)  # type: ignore

    def add(self, predictions: Sequence, labels: Sequence) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to ``self._results``.

        Args:
            predictions (Sequence): Predictions from the model. It can be
                labels (N, ), or scores of every class (N, C).
            labels (Sequence): The ground truth labels. It should be (N, ).
        """
        corrects = self._compute_corrects(predictions, labels)
        for correct in corrects:
            self._results.append(correct)

    def _compute_corrects(
        self, predictions: Union['torch.Tensor', Sequence['torch.Tensor']],
        labels: Union['torch.Tensor',
                      Sequence['torch.Tensor']]) -> 'torch.Tensor':
        """Compute the correct number of per topk and threshold with PyTorch.

        Args:
            prediction (torch.Tensor | Sequence): Predictions from the model.
                Same as ``self.add``.
            labels (torch.Tensor | Sequence): The ground truth labels. Same as
                ``self.add``.

        Returns:
            torch.Tensor: Correct number with the following 2 shapes.

            - (N, ): If the ``predictions`` is a label tensor instead of score.
              Only return a top-1 correct tensor, and ignore the argument
              ``topk`` and ``thrs``.
            - (N, num_topk, num_thr): If the ``prediction`` is a score tensor
              (number of dimensions is 2). Return the correct number on each
              ``topk`` and ``thrs``.
        """
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.stack(predictions)
        if not isinstance(labels, torch.Tensor):
            labels = torch.stack(labels)

        if predictions.ndim == 1:
            corrects = (predictions.int() == labels)
            return corrects.float()

        pred_scores, pred_label = _torch_topk(predictions, self.maxk, dim=1)
        pred_label = pred_label.t()

        corrects = (pred_label == labels.view(1, -1).expand_as(pred_label))

        # compute the corrects corresponding to all topk and thrs per sample
        corrects_per_sample = torch.zeros(
            (len(predictions), len(self.topk), len(self.thrs)))
        for i, k in enumerate(self.topk):
            for j, thr in enumerate(self.thrs):
                # Only prediction socres larger than thr are counted as correct
                if thr is not None:
                    thr_corrects = corrects & (pred_scores.t() > thr)
                else:
                    thr_corrects = corrects
                corrects_per_sample[:, i, j] = thr_corrects[:k].sum(
                    0, keepdim=True).float()
        return corrects_per_sample

    def compute_metric(self,
                       results: List[Union[Iterable, Union[np.number, 'torch.Tensor',]]]
                       ) -> Dict[str, float]:
        """Compute the accuracy metric.

        This method would be invoked in ``BaseMetric.compute`` after
        distributed synchronization.

        Args:
            results (list): A list that consisting the correct numbers. This
                list has already been synced across all ranks.

        Returns:
            Dict[str, float]: The computed accuracy metric.
        """
        if _is_scalar(results[0]):
            return {'top1': float(sum(results) / len(results))}  # type: ignore

        metric_results = {}
        for i, k in enumerate(self.topk):
            for j, thr in enumerate(self.thrs):
                corrects = [result[i][j] for result in results]  # type: ignore
                acc = float(sum(corrects) / len(corrects))
                name = f'top{k}'
                if len(self.thrs) > 1:
                    name += '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                metric_results[name] = acc
        return metric_results


def _torch_topk(inputs: 'torch.Tensor',
                k: int,
                dim: Optional[int] = None) -> Tuple:
    """Invoke the PyTorch topk."""
    return inputs.topk(k, dim=dim)


def _is_scalar(obj):
    """Check if an object is a scalar."""
    try:
        float(obj)  # type: ignore
        return True
    except Exception:
        return False


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=''):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


# Plots ----------------------------------------------------------------------------------------------------------------
def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


# Usage------------------------------------------------------------------
def main():
    labels = torch.Tensor([0, 1, 2, 3])
    preds = torch.Tensor([0, 2, 1, 3])
    test = Accuracy()
    test(preds, labels)

    labels = torch.tensor([0, 1, 2, 3])
    preds = torch.tensor([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.3, 0.4, 0.2],
        [0.3, 0.4, 0.2, 0.1],
        [0.0, 0.0, 0.1, 0.9]])
    test = Accuracy(topk=(1, 2, 3))
    test(preds, labels)


if __name__ == '__main__':
    main()