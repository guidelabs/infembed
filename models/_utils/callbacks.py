from typing import Dict, List, Optional, Tuple
from models._core.decoder_llm import GreedyDecoder
from lightning.pytorch.callbacks import Callback
import wandb
from torchmetrics import MetricCollection


class GenericCallback(Callback):
    """
    callback that runs on end of specified hooks
    """

    def __init__(self, hook_strings: List[str]):
        self.hook_strings = hook_strings

    def on_validation_start(self, trainer, pl_module):
        if "on_validation_start" in self.hook_strings:
            self._on_start(trainer, pl_module, "val")

    def on_train_start(self, trainer, pl_module):
        if "on_train_start" in self.hook_strings:
            self._on_start(trainer, pl_module, "train")

    def on_test_start(self, trainer, pl_module):
        if "on_test_start" in self.hook_strings:
            self._on_start(trainer, pl_module, "test")

    def on_predict_start(self, trainer, pl_module):
        # batch will likely be different than other hooks, so `_on_epoch_end` may not
        # work
        if "on_predict_start" in self.hook_strings:
            self._on_start(trainer, pl_module, "predict")

    def on_validation_epoch_end(self, trainer, pl_module):
        if "on_validation_epoch_end" in self.hook_strings:
            self._on_epoch_end(trainer, pl_module, "val")

    def on_train_epoch_end(self, trainer, pl_module):
        if "on_train_epoch_end" in self.hook_strings:
            self._on_epoch_end(trainer, pl_module, "train")

    def on_test_epoch_end(self, trainer, pl_module):
        if "on_test_epoch_end" in self.hook_strings:
            self._on_epoch_end(trainer, pl_module, "test")

    def on_predict_epoch_end(self, trainer, pl_module):
        # batch will likely be different than other hooks, so `_on_epoch_end` may not
        # work
        if "on_predict_epoch_end" in self.hook_strings:
            self._on_epoch_end(trainer, pl_module, "predict")

    def on_validation_epoch_start(self, trainer, pl_module):
        if "on_validation_epoch_start" in self.hook_strings:
            self._on_epoch_start(trainer, pl_module, "val")

    def on_train_epoch_start(self, trainer, pl_module):
        if "on_train_epoch_start" in self.hook_strings:
            self._on_epoch_start(trainer, pl_module, "train")

    def on_test_epoch_start(self, trainer, pl_module):
        if "on_test_epoch_start" in self.hook_strings:
            self._on_epoch_start(trainer, pl_module, "test")

    def on_predict_epoch_start(self, trainer, pl_module):
        # batch will likely be different than other hooks, so `_on_epoch_end` may not
        # work
        if "on_predict_epoch_start" in self.hook_strings:
            self._on_epoch_start(trainer, pl_module, "predict")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "on_validation_batch_end" in self.hook_strings:
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "val")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "on_train_batch_end" in self.hook_strings:
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "train")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "on_test_batch_end" in self.hook_strings:
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "test")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # batch will likely be different than other hooks, so `_on_epoch_end` may not
        # work
        if "on_predict_batch_end" in self.hook_strings:
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "predict")

    def _on_start(self, trainer, pl_module, phase: str):
        raise NotImplementedError

    def _on_epoch_start(self, trainer, pl_module, phase: str):
        raise NotImplementedError

    def _on_epoch_end(self, trainer, pl_module, phase: str):
        raise NotImplementedError

    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, phase: str):
        raise NotImplementedError


class DecoderCallback(GenericCallback):
    """
    logs a single greedy generation at epoch ends
    """

    def __init__(
        self,
        eos_token_id: int,
        tokenizer,
        hook_strings: List[str],
        max_len: Optional[int] = None,
        num_samples_per_temperature: List[Tuple[float, int]] = {0},
    ):
        GenericCallback.__init__(self, hook_strings)
        self.eos_token_id, self.tokenizer = eos_token_id, tokenizer
        self.max_len = max_len
        self.num_samples_per_temperature = num_samples_per_temperature

    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, phase: str):
        decoder = GreedyDecoder(
            max_len=pl_module.max_len if self.max_len is None else self.max_len
        )
        rows = []
        for temperature, num_samples in self.num_samples_per_temperature:
            for example in batch["input_ids"]:
                for i in range(num_samples):
                    rows.append(
                        [
                            temperature,
                            self.tokenizer.decode(example),
                            self.tokenizer.decode(
                                decoder(
                                    pl_module, self.eos_token_id, example, temperature
                                )
                            ),
                            i,
                        ]
                    )

        table = wandb.Table(
            columns=["temperature", "prompt", "generation", "trial"], data=rows
        )
        # pl_module.log_dict({phase: {'greedy_text': table}})
        wandb.log({"greedy_text": table})


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


class TorchMetricsCallback(GenericCallback):
    def __init__(
        self,
        metrics: Dict[str, torchmetrics.Metric],
        hook_strings: List[str],
        get_preds: Callable = default_get_preds,
        get_target: Callable = default_get_target,
        write_path: str = 'metrics.csv',
        get_preds_and_target: Optional[Callable] = None,
    ):
        # super().__init__(write_interval='batch_and_epoch')
        super().__init__(hook_strings=hook_strings)
        self.metrics = metrics
        self.get_preds, self.get_target = get_preds, get_target
        self.write_path = write_path
        self.get_preds_and_target = get_preds_and_target

    def _on_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        phase,
    ):
        
        if self.get_preds_and_target is None:
            preds = self.get_preds(outputs)
            target = self.get_target(batch)
        else:
            preds, target = self.get_preds_and_target(outputs, batch)

        getattr(pl_module, f"{phase}_metrics")(preds, target)
        # for (_, metric) in self.metrics.items():
        #     val = metric(preds, target)
        #     #metric(preds.cpu(), target.cpu())

    def _on_start(self, trainer, pl_module, phase: str):
        # pl_module.metrics = torchmetrics.MetricCollection(list(self.metrics.values()))
        setattr(pl_module, f"{phase}_metrics", torchmetrics.MetricCollection(dict(self.metrics)))
        #setattr(pl_module, f"{phase}_metrics", torchmetrics.MetricCollection(self.metrics))
        # pl_module.metrics = torchmetrics.MetricCollection(self.metrics)

    def _get_formatted_metrics(self, metric):
        import pdb
        pdb.set_trace()
        return {_metric: float(val) for (_metric, val) in metric.compute()}

    def _on_epoch_end(self, trainer, pl_module, phase: str):
        d = {phase: self._get_formatted_metrics(getattr(pl_module, f"{phase}_metrics"))}


        #d = {f"{phase}_{name}": float(metric.compute()) for (name, metric) in self.metrics.items()}
        import pdb
        pdb.set_trace()
        self.log_dict(d, on_epoch=True)
        # pd.DataFrame({"metric_val": d}).to_csv(open(self.write_path, 'w'))


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