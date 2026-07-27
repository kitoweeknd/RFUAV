# origin: https://github.com/open-mmlab/mmeval
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch

from utils.metrics.base_metric import BaseMetric


class F1Score(BaseMetric):
    """Compute F1 scores.

    Args:
        num_classes (int): Number of labels.
        mode (str or list[str]): There are 2 options:

            - 'micro': Calculate metrics globally by counting the total true
              positives, false negatives and false positives.
            - 'macro': Calculate metrics for each label, and find their
              unweighted mean.

            If mode is a list, then metrics in mode will be calculated
            separately. Defaults to 'micro'.
        cared_classes (list[int]): The indices of the labels participated in
            the metric computing. If both ``cared_classes`` and
            ``ignored_classes`` are empty, all classes will be taken into
            account. Defaults to []. Note: ``cared_classes`` and
            ``ignored_classes`` cannot be specified together.
        ignored_classes (list[int]): The index set of labels that are ignored
            when computing metrics. If both ``cared_classes`` and
            ``ignored_classes`` are empty, all classes will be taken into
            account. Defaults to []. Note: ``cared_classes`` and
            ``ignored_classes`` cannot be specified together.
        **kwargs: Keyword arguments passed to :class:`BaseMetric`.

    Warning:
        Only non-negative integer labels are involved in computing. All
        negative ground truth labels will be ignored.
    """

    def __init__(self,
                 num_classes: int,
                 mode: Union[str, Sequence[str]] = 'micro',
                 cared_classes: Sequence[int] = [],
                 ignored_classes: Sequence[int] = [],
                 **kwargs) -> None:
        super().__init__(**kwargs)

        assert isinstance(num_classes, int)
        assert isinstance(cared_classes, (list, tuple))
        assert isinstance(ignored_classes, (list, tuple))
        assert isinstance(mode, (list, str))
        assert not (len(cared_classes) > 0 and len(ignored_classes) > 0), \
            'cared_classes and ignored_classes cannot be both non-empty'

        if isinstance(mode, str):
            mode = [mode]
        assert set(mode).issubset({'micro', 'macro'})
        self.mode = mode

        if len(cared_classes) > 0:
            assert min(cared_classes) >= 0 and \
                max(cared_classes) < num_classes, \
                'cared_classes must be a subset of [0, num_classes)'
            self.cared_labels = sorted(cared_classes)
        elif len(ignored_classes) > 0:
            assert min(ignored_classes) >= 0 and \
                max(ignored_classes) < num_classes, \
                'ignored_classes must be a subset of [0, num_classes)'
            self.cared_labels = sorted(
                set(range(num_classes)) - set(ignored_classes))
        else:
            self.cared_labels = list(range(num_classes))
        self.cared_labels = np.array(self.cared_labels, dtype=np.int64)
        self.num_classes = num_classes

    def add(self, predictions: Sequence[Union[Sequence[int], np.ndarray]], labels: Sequence[Union[Sequence[int], np.ndarray]]) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Process one batch of data and predictions.

        Calculate the following 2 stuff from the inputs and store them in
        ``self._results``:

        - prediction: prediction labels.
        - label: ground truth labels.

        Args:
            predictions (Sequence[Sequence[int] or np.ndarray]): A batch
                of sequences of non-negative integer labels.
            labels (Sequence[Sequence[int] or np.ndarray]): A batch of
                sequences of non-negative integer labels.
        """
        for prediction, label in zip(predictions, labels):
            self._results.append((prediction, label))

    def _compute_tp_fp_fn(self, predictions: Sequence['torch.Tensor'],
                          labels: Sequence['torch.Tensor']) -> tuple:
        """Compute tp, fp and fn from predictions and labels."""
        preds = torch.cat(predictions).long().flatten().cpu()
        gts = torch.cat(labels).long().flatten().cpu()

        assert preds.max() < self.num_classes
        assert gts.max() < self.num_classes

        cared_labels = preds.new_tensor(self.cared_labels, dtype=torch.long)

        hits = (preds == gts)[None, :]
        preds_per_label = cared_labels[:, None] == preds[None, :]
        gts_per_label = cared_labels[:, None] == gts[None, :]

        tp = (hits * preds_per_label).cpu().numpy().astype(float)
        fp = (~hits * preds_per_label).cpu().numpy().astype(float)
        fn = (~hits * gts_per_label).cpu().numpy().astype(float)
        return tp, fp, fn

    def compute_metric(self, results: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[(ndarray, ndarray)]): The processed results of each
                batch.

        Returns:
            dict[str, float]: The f1 scores. The keys are the names of the
            metrics, and the values are corresponding results. Possible
            keys are 'micro_f1' and 'macro_f1'.
        """

        preds, gts = zip(*results)

        tp, fp, fn = self._compute_tp_fp_fn(preds, gts)

        result = {}
        if 'macro' in self.mode:
            result['macro_f1'] = self._compute_f1(
                tp.sum(-1), fp.sum(-1), fn.sum(-1))
        if 'micro' in self.mode:
            result['micro_f1'] = self._compute_f1(tp.sum(), fp.sum(), fn.sum())

        return result

    def _compute_f1(self, tp: np.ndarray, fp: np.ndarray,
                    fn: np.ndarray) -> float:
        """Compute the F1-score based on the true positives, false positives
        and false negatives.

        Args:
            tp (np.ndarray): The true positives.
            fp (np.ndarray): The false positives.
            fn (np.ndarray): The false negatives.

        Returns:
            float: The F1-score.
        """
        precision = tp / (tp + fp).clip(min=1e-8)
        recall = tp / (tp + fn).clip(min=1e-8)
        f1 = 2 * precision * recall / (precision + recall).clip(min=1e-8)
        return float(f1.mean())


# Usage-------------------------------------------------------------
def main():

    f1 = F1Score(num_classes=5, mode=['macro', 'micro'])

    preds = torch.tensor([
        [0.7, 0.1, 0.1,],
        [0.1, 0.3, 0.4,],
        [0.3, 0.4, 0.2,]])
    _preds = []
    for _ in preds:
        _preds.append(_.argmax())
    preds = [torch.tensor(_preds)]

    labels = [torch.Tensor([0, 1, 4])]

    for pred, label in zip(preds, labels):
        f1.add([pred], [label])

    metric_result = f1.compute_metric(f1._results)
    print(metric_result)



if __name__ == '__main__':
    main()