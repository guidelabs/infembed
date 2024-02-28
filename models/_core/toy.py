from models._utils.common import GenericLightningModule
import torch.nn as nn

"""
this contains functions for the toy model used for testing.
"""

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class ToyLightningModule(GenericLightningModule):

    def _step(self, batch, batch_idx):
        return {"loss": self.loss_fn(self.forward(batch[:-1]), batch[-1])}

    def forward(self, batch):
        return self.decoder(*batch)
