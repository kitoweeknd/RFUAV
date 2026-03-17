from abc import ABCMeta, abstractmethod
import torch
from typing import Any, Dict, List, Optional
from torch import Tensor
from utils.logger import colorful_logger


class BaseMetric(metaclass=ABCMeta):

    """
    Base class for f1, topk, confusionmatrix, precision metric.
    original implementation: https://github.com/open-mmlab/mmeval

    To implement a metric, you should implement a subclass of ``BaseMetric``
    that overrides the ``add`` and ``compute_metric`` methods. ``BaseMetric``
    will automatically complete the distributed synchronization between
    processes.

    During the evaluation process, each metric will update ``self._results``
    to store intermediate results after each call to ``add``. When computing
    the final metric result, the ``self._results`` will be synchronized between
    processes.
    """

    def __init__(self,
                 dataset_meta: Optional[Dict] = None,
                 dist_collect_mode: str = 'unzip',
                 ):
        """Initialize the BaseMetric.

        Args:
            dataset_meta (Optional[Dict], optional): Meta information of the dataset. Defaults to None.
            dist_collect_mode (str, optional): Mode for distributed collection, either 'cat' or 'unzip'. Defaults to 'unzip'.
        """
        self.dataset_meta = dataset_meta
        assert dist_collect_mode in ('cat', 'unzip')
        self.dist_collect_mode = dist_collect_mode
        self._results: List[Any] = []

    @property
    def dataset_meta(self) -> Optional[Dict]:
        """Meta information of the dataset."""
        if self._dataset_meta is None:
            return self._dataset_meta
        else:
            return self._dataset_meta.copy()

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: Optional[Dict]) -> None:
        """Set the dataset meta information to the metric."""
        if dataset_meta is None:
            self._dataset_meta = dataset_meta
        else:
            self._dataset_meta = dataset_meta.copy()

    @property
    def name(self) -> str:
        """The metric name, defaults to the name of the class."""
        return self.__class__.__name__

    def reset(self) -> None:
        """Clear the metric stored results."""
        self._results.clear()

    def __call__(self, *args, **kwargs) -> Dict:
        """Stateless call for a metric compute."""
        cache_results = self._results
        self._results = []
        self.add(*args, **kwargs)
        metric_result = self.compute_metric(self._results)
        self._results = cache_results
        return metric_result


    @abstractmethod
    def add(self, *args, **kwargs):
        """Override this method to add the intermediate results to
        ``self._results``.

        Note:
            For performance issues, what you add to the ``self._results``
            should be as simple as possible. But be aware that the intermediate
            results stored in ``self._results`` should correspond one-to-one
            with the samples, in that we need to remove the padded samples for
            the most accurate result.
        """

    @abstractmethod
    def compute_metric(self, results: List[Any]) -> Dict:
        """Override this method to compute the metric result from collectd
        intermediate results.

        The returned result of the metric compute should be a dictionary.
        """


class EVAMetric:
    def __new__(self,
                preds: Tensor,
                labels: Tensor,
                tasks: tuple[str, ...] = ('f1', 'precision', 'CM'),
                num_classes: Optional[int] = None,
                topk: tuple[int, ...] = None,
                save_path: Optional[str] = None,
                classes_name: Optional[tuple[str, ...]] = None,
                pic_name: str = ''):

        """
        Custom evaluation metrics for predictions and labels.

        Args:
            preds (Tensor): Predicted values.
            labels (Tensor): Ground truth labels.
            tasks (tuple[str, ...], optional): Tasks to perform, e.g., 'f1', 'precision', 'CM'. Defaults to ('f1', 'precision', 'CM').
            num_classes (Optional[int], optional): Number of classes. Defaults to None.
            topk (tuple[int, ...], optional): Top-k values to compute accuracy. Defaults to None.
            save_path (Optional[str], optional): Path to save plots. Defaults to None.
            classes_name (Optional[tuple[str, ...]], optional): Names of the classes. Defaults to None.
            pic_name str : confusion matrix pic name

        Returns:
            dict: Dictionary containing the computed metrics.
        """

        logger = colorful_logger('Evaluate')
        res = {}

        for task in tasks:

            if task == 'f1':
                from .f1 import F1Score
                logger.log_with_color('start computing the f1')
                _preds = []
                for _ in preds:
                    _preds.append(_.argmax())
                _preds = [torch.tensor(_preds)]
                _labels = [labels]
                f1 = F1Score(num_classes=num_classes, mode=['macro', 'micro'])
                for pred, label in zip(_preds, _labels):
                    f1.add([pred], [label])

                res['f1'] = f1.compute_metric(f1._results)

            elif task == 'precision':
                _labels = labels.unsqueeze(1)
                logger.log_with_color('start computing the precision')
                from .precision import AveragePrecision
                ap = AveragePrecision()
                res['mAP'] = ap(preds, labels)

            elif task == 'CM':
                logger.log_with_color('start plotting the confusion matrix')
                from .confusionmatrix import ConfusionMatrix
                cm = ConfusionMatrix(nc=num_classes, pic_name=pic_name)
                cm.process_cls_preds(preds, labels)
                for _ in True, False:
                    cm.plot(normalize=_, save_dir=save_path, names=classes_name)

        from topk import Accuracy
        logger.log_with_color('start computing the Top-k')
        acc = Accuracy(topk=(1, 2, 3))
        res['Top-k'] = acc(preds, labels)

        return res


# Usage-------------------------------------
def main():
    num_images = 100
    num_classes = 5
    save_path = ''
    classes_name = ('A', 'B', 'C', 'D', 'E')

    preds = torch.rand(num_images, num_classes)
    preds = preds / preds.sum(dim=1, keepdim=True)

    labels = torch.randint(0, num_classes, (num_images,))

    metric = EVAMetric(preds=preds,
                       labels=labels,
                       num_classes=5,
                       tasks=('f1', 'precision', 'CM'),
                       topk=(1, 3, 5),
                       save_path=save_path,
                       classes_name=classes_name)

    print(metric)


if __name__ == '__main__':
    main()