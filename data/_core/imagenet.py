from torchvision.models import ResNet18_Weights
from torch.utils.data import default_collate


normalize = ResNet18_Weights.IMAGENET1K_V1.transforms()


class ImagenetCollateFn:
    def __init__(self, device):
        self.device = device

    def __call__(self, examples):
        return tuple(
            [
                _x.to(device=self.device)
                for _x in default_collate(
                    [(normalize(__x[0]), __x[1]) for __x in examples]
                )
            ]
        )
