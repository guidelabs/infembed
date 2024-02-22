from typing import List
from models._utils.common import clones
import torch.nn as nn
from models._utils.common import MLP, clones
import torch.nn as nn
import torch.nn.functional as F
import torch
from models._utils.callbacks import GenericCallback
from models._utils.callbacks import MultitaskTorchMetricsCallback, TorchMetricsCallback
from torchmetrics import MetricCollection, AUROC
from torchmetrics.wrappers import MultitaskWrapper
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score


class LLMBinaryMultitaskLoss(nn.Module):
    def __init__(self, equal_batch_contribution: bool = True):
        super().__init__()
        self.equal_batch_contribution = equal_batch_contribution

    def forward(self, prediction_logits, attention_mask, labels):
        """
        this loss is generic, but specialized to the CB case, the arguments are as
        follows:
        `prediction_logits` shape: batch size X sequence length X number concepts
        `labels` shape: batch size X sequence length X number concepts
        interpretation of `concept_labels[i,t,c]`: whether for example i, concept c
        exists in x[0:i+1], i.e. the prefix ending at and including token i.

        TODO: to get `concept_labels`, we will train a token-level multi-label classifier
        and apply to each token for each concept.  training this would be non-standard,
        because we will only assume we know whether a concept is present in a sequence, but
        not which prefixes it is present in, i.e. we don't have token-level labels.  this is
        a case of learning a classifier with ambiguous labels.  to start simple, for each sequence
        for which a concept is present, we can assume the concept is present in all prefixes,
        i.e. all token-level labels for that concept are positive.  this should actually do
        okay, because models can handle noisy labels.
        """

        def _loss(_prediction_logits, _labels, _attention_mask):
            return (
                F.binary_cross_entropy_with_logits(
                    _prediction_logits, _labels.float(), reduction="none"
                )
                * _attention_mask[..., None]
            ).sum() / (
                torch.sum(_attention_mask) * _prediction_logits.shape[1]
            )  # scale by number of tasks so that loss is average per task

        if not self.equal_batch_contribution:
            return _loss(prediction_logits, labels, attention_mask)
        else:
            return torch.mean(
                torch.stack(
                    [
                        _loss(_prediction_logits, _labels, _attention_mask)
                        for (_prediction_logits, _labels, _attention_mask) in zip(
                            prediction_logits, labels, attention_mask
                        )
                    ]
                )
            )


class LLMMILImputeLoss(
    nn.Module
):  # TODO: move loss functions to own file, since they are independent of model
    """
    for the LLM MIL scenario (have per-example and optionally, per-token labels), imputes
    the per-token labels to be the same as the per-example label, and applies a provided
    per-token loss
    """

    def __init__(self, llm_loss):
        super().__init__()
        self.llm_loss = llm_loss

    def forward(self, prediction_logits, attention_mask, labels, example_labels):
        _labels = []
        for example_label in example_labels:
            __labels = torch.zeros(prediction_logits.shape[1:3]).to(
                device=prediction_logits.device
            )
            _labels.append(__labels.masked_fill(example_label[None, :].bool(), 1))
        _labels = torch.stack(_labels, dim=0)
        return self.llm_loss(
            prediction_logits, attention_mask, _labels.requires_grad_()
        )
    

class LLMMILExampleLoss(
    nn.Module
):  # TODO: move loss functions to own file, since they are independent of model
    """
    for the LLM MIL scenario (have per-example and optionally, per-token labels), uses
    only the predictions for the last position and the per-example labels
    """

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, prediction_logits, attention_mask, labels, example_labels):
        example_logits = _last_token_multitask_get_preds(prediction_logits, attention_mask)
        return self.loss(example_logits, example_labels.float())


class LLMMilMaxLoss(nn.Module):
    """
    for the LLM MIL scenario (have per-example and optionally, per-token labels), takes
    the max of the per-token predictions to get a per-example prediction, and gives it
    to a provided per-example loss.
    TODO: can try a leaky max or other aggregation function
    """

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, prediction_logits, attention_mask, labels, example_labels):
        _prediction_logits = prediction_logits.masked_fill(attention_mask.bool(), -1e9)
        _prediction_logits = torch.max(_prediction_logits, dim=1)
        return self.loss(_prediction_logits, example_labels)


class LLMBinaryMultitaskMLPGenerator(nn.Module):
    """
    maps from sequence of embeddings (i.e. decoder output) to scalar predictions for
    each of several tasks
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_tasks,
        pre_nonlinearity,
        post_nonlinearity,
        task_specific_dim=True,
    ):
        super().__init__()
        self.mlps = clones(
            MLP(
                [input_dim] + hidden_dims + [1],
                pre_nonlinearity=pre_nonlinearity,
                post_nonlinearity=post_nonlinearity,
            ),
            num_tasks,
        )
        self.task_specific_dim = task_specific_dim

    def forward(self, x):
        # for each mlp: sequence length X number concepts X input dim -> sequence length X number concepts
        # for concept embedding, input dim will be 2 * concept embedding dim
        if self.task_specific_dim:
            # each task gets own dimension / input, so input is 4D
            assert x.shape[2] == len(self.mlps)
            return torch.cat(
                [self.mlps[c](x[:, :, c, :]) for c in range(len(self.mlps))], dim=2
            )
        else:
            # each task shares the input, so input is 3D
            return torch.cat([self.mlps[c](x) for c in range(len(self.mlps))], dim=2)


def multitask_get_preds(outputs):
    return outputs["prediction_logits"]


def _last_token_multitask_get_preds(preds, attention_mask):
    # preds = multitask_get_preds(outputs)
    # attention_mask = batch["attention_mask"]
    last_token = (
        attention_mask.shape[1]
        - 1
        - torch.argmax(torch.flip(attention_mask, dims=[1]).float(), dim=1)
    ).long()
    assert torch.sum(last_token < 0) == 0
    return torch.stack(
        [
            torch.gather(preds[:, :, t], 1, last_token[:, None]).squeeze()
            for t in range(preds.shape[2])
        ],
        dim=1,
    )


def last_token_multitask_get_preds_and_target(outputs, batch):
    """
    this is a model that can only output per-token predictions, but we want to measure
    per-example performance.  so need to get a per-example prediction.  here, we use
    the prediction for the last token as that.
    """
    preds = {
        f"task_{t}": task_preds.detach().cpu()
        for (t, task_preds) in enumerate(
            _last_token_multitask_get_preds(multitask_get_preds(outputs), batch["attention_mask"]).T
        )
    }

    target = batch["example_labels"]
    target = {f"task_{t}": target[:, t].detach().cpu() for t in range(target.shape[1])}
    # TODO: put metrics on right device so don't have to move preds and target to cpu
    return preds, target


def example_labels_metrics_callback(num_tasks, get_preds_and_target=last_token_multitask_get_preds_and_target):
    return MultitaskTorchMetricsCallback(
        metrics=[BinaryAccuracy(), AUROC(task="binary")],
        hook_strings=[
            "on_validation_start",
            "on_validation_epoch_end",
            "on_validation_batch_end",
        ],
        num_tasks=num_tasks,
        get_preds_and_target=get_preds_and_target,
    )


class LLMBinaryMultitaskWritePredictions(GenericCallback):
    """
    writes predictions as a file which is csv format with columns example index, then
    a column for the predictions for each task
    """

    def __init__(self, hook_strings: List[str], filename="predictions"):
        GenericCallback.__init__(self, hook_strings)
        self.started_writing = False
        self.i = 0
        self.filename = filename

    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, phase: str):

        path = f"{phase}_{self.filename}.csv"
        preds = multitask_get_preds(outputs)

        # write the header
        if not self.started_writing:
            num_tasks = next(iter(preds)).shape[1]
            with open(path, "w") as f:
                f.write(
                    ",".join(["i", "t"] + [f"task_{k}" for k in range(num_tasks)])
                    + "\n"
                )
            self.started_writing = True

        # write the predictions
        with open(path, "a") as f:
            for pred in preds:
                for t, _pred in enumerate(pred):
                    f.write(
                        ",".join(
                            [str(self.i), str(t)] + list(map(str, map(float, _pred)))
                        )
                        + "\n"
                    )
                self.i += 1
