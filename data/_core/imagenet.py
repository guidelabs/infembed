from torchvision.models import ResNet18_Weights
from torch.utils.data import default_collate


normalize = ResNet18_Weights.IMAGENET1K_V1.transforms()


def get_collate_fn(device):
    def collate_fn(examples):
        return tuple(
            [
                _x.to(device=device)
                for _x in default_collate(
                    [(normalize(__x[0]), __x[1]) for __x in examples]
                )
            ]
        )

    return collate_fn
