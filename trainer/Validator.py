import torch
from ignite.engine import Engine
from ignite.metrics import PSNR, Precision, Recall


def eval_step(engine, batch, apply_threshold=True, threshold=0.5):
    inputs, targets = batch
    if apply_threshold:
        inputs = torch.where(inputs > threshold, 1., 0.)
        targets = torch.where(targets > threshold, 1., 0.)
    inputs = 1.0 - inputs
    targets = 1.0 - targets

    return inputs, targets


class Validator:
    def __init__(self, apply_threshold=True, threshold=0.5):
        self._evaluator = Engine(eval_step)
        self.apply_threshold = apply_threshold
        self.threshold = threshold

        self._psnr = PSNR(data_range=1.0)
        self._psnr.attach(self._evaluator, 'psnr')

        self._precision = Precision()
        self._precision.attach(self._evaluator, 'precision')

        self._recall = Recall()
        self._recall.attach(self._evaluator, 'recall')

        self._count = 0
        self._psnr_value = 0.0
        self._precision_value = 0.0
        self._recall_value = 0.0

    def compute(self, predicts: torch.Tensor, targets: torch.Tensor):
        state = self._evaluator.run([[predicts, targets]], self.apply_threshold, self.threshold)

        self._count += len(predicts)
        self._psnr_value += state.metrics['psnr']
        self._precision_value += state.metrics['precision']
        self._recall_value += state.metrics['recall']

        avg_psnr = state.metrics['psnr'] / len(predicts)
        avg_precision = 100. * state.metrics['precision'] / len(predicts)
        avg_recall = 100. * state.metrics['recall'] / len(predicts)

        return avg_psnr, avg_precision, avg_recall

    def get_metrics(self):
        psnr = self._psnr_value / self._count
        precision = 100. * self._precision_value / self._count
        recall = 100. * self._recall_value / self._count
        return psnr, precision, recall

    def reset(self):
        self._count = 0
        self._psnr_value = 0
        self._precision_value = 0
        self._recall_value = 0
