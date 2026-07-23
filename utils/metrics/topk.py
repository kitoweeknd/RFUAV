"""
origin: https://github.com/open-mmlab/mmeval
"""
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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

        # torch.topk requires k <= num_classes. Clamp so callers requesting a
        # top-k larger than the number of classes (e.g. top-5 on a 3-class model)
        # don't crash; for k >= num_classes every sample is trivially "correct",
        # which the `thr_corrects[:k]` slice below already yields.
        num_classes = predictions.shape[-1]
        safe_maxk = min(self.maxk, num_classes)
        pred_scores, pred_label = _torch_topk(predictions, safe_maxk, dim=1)
        pred_label = pred_label.t()

        corrects = (pred_label == labels.view(1, -1).expand_as(pred_label))

        # compute the corrects corresponding to all topk and thrs per sample
        corrects_per_sample = torch.zeros(
            (len(predictions), len(self.topk), len(self.thrs)),
            device=predictions.device)
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