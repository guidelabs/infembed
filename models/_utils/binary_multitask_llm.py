from models._utils.common import clones
import torch.nn as nn
from models._utils.common import MLP, clones
import torch.nn as nn
import torch.nn.functional as F
import torch


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
                    _prediction_logits, _labels, reduction="none"
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
        _labels = torch.zeros(prediction_logits.shape, requires_grad=True)
        for __labels, example_label in zip(_labels, example_labels):
            __labels.masked_fill(example_label[None, :].bool(), 1)
        return self.llm_loss(
            prediction_logits, attention_mask, _labels.requires_grad_()
        )


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


def last_token_get_preds_and_target(outputs, batch):
    """
    this is a model that can only output per-token predictions, but we want to measure
    per-example performance.  so need to get a per-example prediction.  here, we use
    the prediction for the last token as that.
    """
    preds = outputs["prediction_logits"]
    last_token = torch.argmin(batch["attention_mask"] == 0, dim=1) - 1
    assert torch.sum(last_token < 0) == 0
    preds = {f"task_{t}": torch.gather(preds[:,:,t], 1, last_token[:, None]) for t in range(preds.shape[2])}
    # preds = torch.gather(preds, 1, last_token[:, None])
    target = batch['example_labels']
    target = {f"task_{t}": target[:,t] for t in range(target.shape[1])}
    return preds, target


from models._utils.callbacks import TorchMetricsCallback
from torchmetrics import MetricCollection
from torchmetrics.wrappers import MultitaskWrapper


def example_labels_metrics_callback():
    return TorchMetricsCallback(metrics={
        
    })
