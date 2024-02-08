from typing import Callable, Dict, List
import torch
# from pytorch_lightning.callbacks import BasePredictionWriter
from lightning.pytorch.callbacks import BasePredictionWriter
import torchmetrics
import pandas as pd
import os


def default_get_preds(out):
    return out.cpu()


def default_get_target(batch):
    return batch[-1].cpu()


class TorchMetricsCallback(BasePredictionWriter):
    def __init__(
        self,
        metrics: Dict[str, torchmetrics.Metric],
        get_preds: Callable = default_get_preds,
        get_target: Callable = default_get_target,
        write_path: str = 'metrics.csv',
    ):
        super().__init__(write_interval='batch_and_epoch')
        self.metrics = metrics
        self.get_preds, self.get_target = get_preds, get_target
        self.write_path = write_path

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        preds = self.get_preds(prediction)
        target = self.get_target(batch)

        for (_, metric) in self.metrics.items():
            metric(preds.cpu(), target.cpu())
            #metric(preds.cpu(), target.cpu())

    def on_predict_start(self, trainer, pl_module):
        # pl_module.metrics = torchmetrics.MetricCollection(list(self.metrics.values()))
        pl_module.metrics = torchmetrics.MetricCollection(self.metrics)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        d = {name: float(metric.compute()) for (name, metric) in self.metrics.items()}
        pd.DataFrame({"metric_val": d}).to_csv(open(self.write_path, 'w'))


class BatchEndWriter(BasePredictionWriter):
    """
    takes extractor functions of the batch or predictions and writes them to file
    """
    def __init__(self, batch_extractors_d: Dict = {}, prediction_extractors_d: Dict = {}):
        super().__init__(write_interval='batch')
        self.batch_extractors_d = batch_extractors_d
        self.prediction_extractors_d = prediction_extractors_d

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        for (name, batch_extractor) in self.batch_extractors_d.items():
            try:
                os.makedirs(name)
            except:
                pass
            extracted = batch_extractor(batch)
            if not isinstance(extracted, pd.DataFrame):
                extracted = pd.DataFrame(extracted)
            extracted.to_csv(f"{name}/{batch_idx}.csv")

        for (name, prediction_extractor) in self.prediction_extractors_d.items():
            try:
                os.makedirs(name)
            except:
                pass
            extracted = prediction_extractor(prediction)
            if not isinstance(extracted, pd.DataFrame):
                extracted = pd.DataFrame(extracted)
            extracted.to_csv(f"{name}/{batch_idx}.csv")