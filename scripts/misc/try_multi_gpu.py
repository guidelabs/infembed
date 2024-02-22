import functools
from data._utils.common import GenericDataModule
from data._core.toy import ToyDataset
from models._core.toy import ToyModel
import lightning as L
from models._utils.common import GenericConfigureOptimizers, GenericLightningModel, get_all_parameters
import torch
from torch.utils.data import DataLoader


def run():
    module = GenericLightningModel(
        model=ToyModel(),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"),
        configure_optimizers=GenericConfigureOptimizers(
            parameters_getter=get_all_parameters,
            optimizer_constructor=functools.partial(torch.optim.Adam, lr=1e-3),
        ),
    )

    datamodule = GenericDataModule(
        train_dataloader=DataLoader(ToyDataset(), batch_size=5, num_workers=0),
        val_dataloader=DataLoader(ToyDataset(), batch_size=5, num_workers=0),
    )

    trainer = L.Trainer(strategy='ddp', devices=[0,1])

    trainer.fit(module, datamodule)


if __name__ == '__main__':
    run()