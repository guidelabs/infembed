import torch.nn.functional as F
import torch.nn as nn

"""
this contains functions needed for the agnews model
"""


class AGNewsLoss(nn.Module):
    def __init__(self, reduction: str):
        super().__init__()
        self.reduction = reduction

    def forward(self, out, batch):
        return F.cross_entropy(out["logits"], batch["labels"], reduction=self.reduction)


def get_preds(out):
    return out["logits"]
