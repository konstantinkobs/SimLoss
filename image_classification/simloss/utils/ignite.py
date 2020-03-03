import logging
import numpy as np
import pandas as pd
import torch
from ignite.metrics.metric import Metric

from simloss.utils.constants import CIFAR100_SUPERCLASSES, CIFAR100_CLASSES_FILTERED

# Abuse ignite metric to save predictions
# NOTE: THIS DOES NOT WORK WHEN SHUFFLE=TRUE BECAUSE PREDICTIONS GO TO THE WRONG DATA POINT!!!
class PredictionMetric(Metric):
    def __init__(self, labels, output_file):
        super(PredictionMetric, self).__init__()
        self._labels = labels
        self._predictions = np.array([])
        self._output_file = output_file

    def update(self, output):
        self._predictions = torch.max(output[0], dim=1)[1]
        self._output = output
        self._class_probs = {}
        for i in range(output[0].shape[1]):
            self._class_probs[f'class_prob_{i}'] = output[0][:, i]

    def compute(self):
        logger = logging.getLogger(__name__)
        logger.info('Writing predictions')
        # Write the predictions
        data = pd.DataFrame({**{'label': self._output[1], 'prediction': self._predictions}, **self._class_probs})
        data.to_csv(self._output_file, index=False)
        return 0

    def reset(self):
        return 0


class SuperclassAccuracy(Metric):
    """Measure the accuracy of predicting the super class indirectly"""
    def __init__(self):
        super(SuperclassAccuracy, self).__init__()
        self._correct = 0
        self._length = 0

    def update(self, output):
        y_pred, y = output
        y_pred = torch.max(y_pred, dim=1)[1]

        y_pred_sc = [CIFAR100_SUPERCLASSES[CIFAR100_CLASSES_FILTERED[p]] for p in y_pred.data.numpy()]
        y_sc = [CIFAR100_SUPERCLASSES[CIFAR100_CLASSES_FILTERED[p]] for p in y.data.numpy()]
        self._correct += sum([y_p == y_t for y_p, y_t in zip(y_pred_sc, y_sc)])
        self._length += len(y)

    def compute(self):
        return self._correct / self._length

    def reset(self):
        self._correct = 0
        self._length = 0


class FailedSuperclassAccuracy(Metric):
    """Only measure the super class accuracy for wrongly classified examples"""
    def __init__(self):
        super(FailedSuperclassAccuracy, self).__init__()
        self._correct = 0
        self._length = 0

    def update(self, output):
        y_pred, y = output
        y_pred = torch.max(y_pred, dim=1)[1]

        class_correct = sum([p == t for p, t in zip(y_pred.data.numpy(), y.data.numpy())])
        y_pred_sc = [CIFAR100_SUPERCLASSES[CIFAR100_CLASSES_FILTERED[p]] for p in y_pred.data.numpy()]
        y_sc = [CIFAR100_SUPERCLASSES[CIFAR100_CLASSES_FILTERED[p]] for p in y.data.numpy()]
        superclass_correct = sum([y_p == y_t for y_p, y_t in zip(y_pred_sc, y_sc)])
        
        self._correct += superclass_correct - class_correct
        self._length += len(y) - class_correct

    def compute(self):
        return self._correct / self._length

    def reset(self):
        self._correct = 0
        self._length = 0
