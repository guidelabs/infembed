from models._utils.common import MLP, clones
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