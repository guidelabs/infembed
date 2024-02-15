from infembed.models._utils.common import MLP, clones
import torch.nn as nn
import torch.nn.functional as F
import torch


"""
this file stores code applicable to all LLM's / sequence models
"""


class LLMCrossEntropyLoss(nn.Module):
    """
    for sequence data, computes cross entropy loss.  currently averages over all
    positions, instead of having each batch contribute equally
    """
    def forward(self, prediction_logits, attention_mask, labels):
        # get per-example, per-position losses
        # losses = F.cross_entropy(output, labels, reduction='none')
        losses = F.cross_entropy(
            prediction_logits.reshape(-1, prediction_logits.shape[-1]), labels.reshape(-1), reduction="none"
        )
        if False:
            brute = sum(
                [
                    F.cross_entropy(_output, _labels, reduction="sum")
                    for (_output, _labels) in zip(output, labels)
                ]
            )
            print(losses.sum(), brute)

        # multiply by attention mask to ignore padding locations
        # losses *= attention_mask
        losses *= attention_mask.reshape(-1)
        # sum up losses over non-padding locations
        loss = losses.sum()
        # divide by total number of non-padding locations
        loss /= attention_mask.sum()
        return loss
    

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

    def forward(self, x):
        # sequence length X number concepts X 2 * concept embedding dim -> sequence length X number concepts
        return torch.cat(
            [self.mlps[c](x[:, :, c, :]) for c in range(x.shape[2])], dim=2
        )
    

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
                * _attention_mask[:, None]
            ).sum() / (
                torch.sum(_attention_mask) * _prediction_logits.shape[1]
            )  # scale by number of tasks so that loss is average per task

        if not self.equal_batch_contribution:
            return _loss(prediction_logits, labels, attention_mask)
        else:
            return torch.mean(
                torch.Tensor(
                    [
                        _loss(_prediction_logits, _labels, _attention_mask)
                        for (_prediction_logits, _labels, _attention_mask) in zip(
                            prediction_logits, labels, attention_mask
                        )
                    ]
                )
            )