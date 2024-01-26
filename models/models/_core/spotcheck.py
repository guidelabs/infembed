from torchvision.models import resnet18
import torch


def get_spotcheck_model(checkpoint_path, device):
    model = resnet18()
    model.fc = torch.nn.Linear(in_features=512, out_features=1)
    model.load_state_dict(torch.load(open(checkpoint_path, "rb"), map_location=device))
    model.to(device=device)
    model.eval()
    return model