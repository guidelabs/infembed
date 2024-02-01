from torchvision.models import resnet18
import torch
from models._utils.common import init_linear
import torchvision
import torch.nn.functional as F
import lightning.pytorch as pl


def get_spotcheck_model(checkpoint_path, device):
    model = resnet18()
    model.fc = torch.nn.Linear(in_features=512, out_features=1)
    model.load_state_dict(torch.load(open(checkpoint_path, "rb"), map_location=device))
    model.to(device=device)
    model.eval()
    return model


class SpotcheckLightningModule(pl.LightningModule):

    def __init__(self, configure_optimizers):
        super().__init__()
        self.model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)
        self._configure_optimizers = configure_optimizers
        self.model.fc.apply(init_linear)
        
    def configure_optimizers(self):
        self._configure_optimizers(self)
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, x):
        return self.model(x)
        
    def _step(self, batch, batch_idx):
        # run forward
        x, y = batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction='mean')
        return {'loss': loss}
    
    def training_step(self, batch, batch_idx):
        d = self._step(batch, batch_idx)
        self.log_dict({f"train_{key}": val for (key, val) in d.items() if key[0] != '_'})
        return d
    
    def validation_step(self, batch, batch_idx):
        d = self._step(batch, batch_idx)
        self.log_dict({f"validation_{key}": val for (key, val) in d.items() if key[0] != '_'})
        return d

    def prediction_step(self, batch, batch_idx):
        d = self._step(batch, batch_idx)
        self.log_dict({f"prediction_{key}": val for (key, val) in d.items() if key[0] != '_'})
        return d
