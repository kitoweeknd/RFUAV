# origin: https://github.com/open-mmlab/mmeval
import warnings
from typing import Dict, List, Optional, Sequence, Union

import torch

from utils.metrics.base_metric import BaseMetric


class MultiLabelMixin:
    """A Mixin for Multilabel Metrics to clarify whether the input is one-hot
    encodings or label-format inputs for corner case with minimal user
    awareness."""

    def __init__(self, *args, **kwargs) -> None:
        # pass arguments for multiple inheritances
        super().__init__(*args, **kwargs)  # type: ignore
        self._pred_is_onehot: Optional[bool] = None
        self._label_is_onehot: Optional[bool] = None

    @property
    def pred_is_onehot(self) -> Optional[bool]:
        """Whether prediction is one-hot encodings.

        Only needed for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        return self._pred_is_onehot

    @pred_is_onehot.setter
    def pred_is_onehot(self, is_onehot: Optional[bool]):
        """Set a flag of whether prediction is one-hot encodings.

        Only needed for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        self._pred_is_onehot = is_onehot

    @property
    def label_is_onehot(self) -> Optional[bool]:
        """Whether label is one-hot encodings.

        Only needed for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        return self._label_is_onehot

    @label_is_onehot.setter
    def label_is_onehot(self, is_onehot: Optional[bool]):
        """Set a flag of whether label is one-hot encodings.

        Only needed for corner case when num_classes=2 to distinguish one-hot
        encodings or label-format.
        """
        self._label_is_onehot = is_onehot


class AveragePrecision(MultiLabelMixin, BaseMetric):
    """Calculate the average precision with respect of classes.

    Args:
        average (str, optional): The average method. It supports two modes:

            - `"macro"`: Calculate metrics for each category, and calculate
                the mean value over all categories.
            - `None`: Return scores of all categories.

        Defaults to "macro".

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`_
    """

    def __init__(self, average: Optional[str] = 'macro', **kwargs) -> None:
        super().__init__(**kwargs)
        average_options = ['macro', None]
        assert average in average_options, 'Invalid `average` argument, ' \
            f'please specify from {average_options}.'
        self.average = average
        self.pred_is_onehot = False

    def add(self, preds: Sequence, labels: Sequence) -> None:  # type: ignore # yapf: disable # noqa: E501
        """Add the intermediate results to `self._results`.

        Args:
            preds (Sequence): Predictions from the model. It should
                be scores of every class (N, C).
            labels (Sequence): The ground truth labels. It should be (N, ) for
                label-format, or (N, C) for one-hot encoding.
        """
        for pred, target in zip(preds, labels):
            self._results.append((pred, target))

    def _format_metric_results(self, ap):
        """Format the given metric results into a dictionary.

        Args:
            ap (list): Results of average precision for each categories
                or the single marco result.

        Returns:
            dict: The formatted dictionary.
        """
        result_metrics = dict()

        if self.average is None:
            _result = ap[0].tolist()
            result_metrics['AP_classwise'] = [round(_r, 4) for _r in _result]
        else:
            result_metrics['mAP'] = round(ap.item(), 4)

        return result_metrics

    def _compute_metric(self, preds: Sequence['torch.Tensor'],
                        labels: Sequence['torch.Tensor']) -> List[List]:
        """A PyTorch implementation that computes the metric."""

        preds = torch.stack(preds)
        num_classes = preds.shape[1]
        labels = format_data(labels, num_classes, self._label_is_onehot).long()

        assert preds.shape[0] == labels.shape[0], \
            'Number of samples does not match between preds' \
            f'({preds.shape[0]}) and labels ({labels.shape[0]}).'

        return _average_precision_torch(preds, labels, self.average)

    def compute_metric(self, results) -> Dict[str, float]:
        """Compute the metric.

        Currently, there are 3 implementations of this method: NumPy and
        PyTorch and OneFlow. Which implementation to use is determined by the
        type of the calling parameters. e.g. `numpy.ndarray` or
        `torch.Tensor`, `oneflow.Tensor`.

        This method would be invoked in `BaseMetric.compute` after distributed
        synchronization.

        Args:
            results (List[Union[NUMPY_IMPL_HINTS, TORCH_IMPL_HINTS,
            ONEFLOW_IMPL_HINTS]]): A list of tuples that consisting the
            prediction and label. This list has already been synced across
            all ranks.

        Returns:
            Dict[str, float]: The computed metric.
        """
        preds = [res[0] for res in results]
        labels = [res[1] for res in results]
        assert self._pred_is_onehot is False, '`self._pred_is_onehot` should' \
            f'be `False` for {self.__class__.__name__}, because scores are' \
            'necessary for compute the metric.'
        metric_results = self._compute_metric(preds, labels)
        return self._format_metric_results(metric_results)


def format_data(
    data: Union[Sequence[Union['torch.Tensor']], 'torch.Tensor'],
    num_classes: int,
    is_onehot: Optional[bool] = None
) -> Union['torch.Tensor']:
    """Format data from different inputs such as prediction scores, label-
    format data and one-hot encodings into the same output shape of `(N,
    num_classes)`.

    Args:
        data (Union[Sequence[np.ndarray, 'torch.Tensor', 'oneflow.Tensor'],
        np.ndarray, 'torch.Tensor', 'oneflow.Tensor']):
            The input data of prediction or labels.
        num_classes (int): The number of classes.
        is_onehot (bool): Whether the data is one-hot encodings.
            If `None`, this will be automatically inducted.
            Defaults to `None`.

    Return:
        torch.Tensor or oneflow.Tensor or np.ndarray:
        One-hot encodings or predict scores.
    """
    if torch and isinstance(data[0], torch.Tensor):
        stack_func = torch.stack
    else:
        raise NotImplementedError(f'Data type of {type(data[0])}'
                                  'is not supported.')

    def _induct_is_onehot(inferred_data):
        """Conduct the input data format."""
        shapes = {d.shape for d in inferred_data}
        if len(shapes) == 1:
            # stack scores or one-hot indices directly if have same shapes
            cand_formated_data = stack_func(inferred_data)
            # all the conditions below is to find whether labels that are
            # raw indices which should be converted to one-hot indices.
            # 1. one-hot indices should has 2 dims;
            # 2. one-hot indices should has num_classes as the second dim;
            # 3. one-hot indices values should always smaller than 2.
            if cand_formated_data.ndim == 2 \
                and cand_formated_data.shape[1] == num_classes \
                    and cand_formated_data.max() <= 1:
                if num_classes > 2:
                    return True, cand_formated_data
                elif num_classes == 2:
                    # 4. corner case, num_classes=2, then one-hot indices
                    # and raw indices are undistinguishable, for instance:
                    #   [[0, 1], [0, 1]] can be one-hot indices of 2 positives
                    #   or raw indices of 4 positives.
                    # Extra induction is needed.
                    warnings.warn(
                        'Ambiguous data detected, reckoned as scores'
                        ' or label-format data as defaults. Please set '
                        'parms related to `is_onehot` to `True` if '
                        'use one-hot encoding data to compute metrics.')
                    return False, None
                else:
                    raise ValueError(
                        'num_classes should greater than 2 in multi label'
                        'metrics.')
        return False, None

    formated_data = None
    if is_onehot is None:
        is_onehot, formated_data = _induct_is_onehot(data)

    if not is_onehot:
        # convert label-format inputs to one-hot encodings
        formated_data = stack_func(
            [label_to_onehot(sample, num_classes) for sample in data])
    elif is_onehot and formated_data is None:
        # directly stack data if `is_onehot` is set to True without induction
        formated_data = stack_func(data)

    return formated_data


def _average_precision_torch(preds: 'torch.Tensor',
                             labels: 'torch.Tensor', average=None) -> 'torch.Tensor':
    r"""Calculate the average precision for torch.

    AP summarizes a precision-recall curve as the weighted mean of maximum
    precisions obtained for any r'>r, where r is the recall:

    .. math::
        \text{AP} = \sum_n (R_n - R_{n-1}) P_n

    Note that no approximation is involved since the curve is piecewise
    constant.

    Args:
        preds (torch.Tensor): The model prediction with shape
            ``(N, num_classes)``.
        labels (torch.Tensor): The target of predictions with shape
            ``(N, num_classes)``.

    Returns:
        torch.Tensor: average precision result.
    """
    # sort examples along classes
    sorted_pred_inds = torch.argsort(preds, dim=0, descending=True)
    sorted_target = torch.gather(labels, 0, sorted_pred_inds)

    # get indexes when gt_true is positive
    pos_inds = sorted_target == 1

    # Calculate cumulative tp case numbers
    tps = torch.cumsum(pos_inds, 0)
    total_pos = tps[-1].clone()  # the last of tensor may change later

    # Calculate cumulative tp&fp(pred_poss) case numbers
    pred_pos_nums = torch.arange(1, len(sorted_target) + 1).to(preds.device)

    tps[torch.logical_not(pos_inds)] = 0
    precision = tps / pred_pos_nums.unsqueeze(-1).float()  # divide along rows
    ap = torch.sum(precision, 0) / torch.clamp(total_pos, min=1)

    if average == 'macro':
        return ap.mean() * 100.0
    else:
        return ap * 100


def label_to_onehot(label: Union['torch.Tensor'], num_classes: int) -> Union['torch.Tensor']:
    """Convert the label-format input to one-hot encodings.

    Args:
        label (torch.Tensor or oneflow.Tensor or np.ndarray):
            The label-format input. The format of item must be label-format.
        num_classes (int): The number of classes.

    Return:
        torch.Tensor or oneflow.Tensor or np.ndarray:
        The converted one-hot encodings.
    """
    if torch and isinstance(label, torch.Tensor):
        label = label.long()
        onehot = label.new_zeros((num_classes, ))

    assert label.max().item() < num_classes, \
        'Max index is out of `num_classes` {num_classes}'
    assert label.min().item() >= 0
    onehot[label] = 1
    return onehot


# Usage----------------------------------------------------------------------------
def main():

    # preds is the probability distribution of each class.
    preds = torch.Tensor([[0.9, 0.8, 0.3, 0.2],
                          [0.1, 0.2, 0.2, 0.1],
                          [0.7, 0.5, 0.9, 0.3],
                          [0.8, 0.1, 0.1, 0.2]])

    # labels is the class of image, also can be presented in one-hot code.
    labels = torch.Tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    labels = torch.Tensor([[0], [1], [2], [3]])

    average_precision = AveragePrecision()
    print(average_precision(preds, labels))


if __name__ == '__main__':
    main()