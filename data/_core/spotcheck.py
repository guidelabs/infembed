from torchvision.datasets import VisionDataset
from torchvision import transforms
from torchvision.transforms.functional import pad
import torch, json
from PIL import Image
import numpy as np
from collections import defaultdict
import pandas as pd


def _get_padding(image):
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class _MakeSquare(object):
    def __call__(self, img):
        return pad(img, _get_padding(img), 0, "constant")

    def __repr__(self):
        return self.__class__.__name__


def _get_transform(mode="normalize"):
    if mode == "normalize":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif mode == "reshape":
        return transforms.Compose([_MakeSquare(), transforms.Resize((224, 224))])
    elif mode == "full":
        return transforms.Compose(
            [
                _MakeSquare(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif mode == "imagenet":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif mode == "resize-crop":
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )


class _ImageDataset(VisionDataset):
    def __init__(
        self,
        filenames,
        labels,
        transform_mode="normalize",
        get_names=False,
    ):
#        print(1)
        transform = _get_transform(mode=transform_mode)
#        print(2)
        super(_ImageDataset, self).__init__(None, None, transform, None)
#        print(3)
        self.filenames = filenames
        self.labels = labels
        self.get_names = get_names
#        print(4)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        if self.get_names:
            return img, label, filename
        else:
            return img, label

    def __len__(self):
        return len(self.filenames)


def _load_data(ids, images, data_key="file", index=None, file_name_replace_prefix_from=None, file_name_replace_prefix_to=None):
    files = []
    labels = []
    for i in ids:
        name = images[i][data_key]
        if file_name_replace_prefix_from is not None:
            name = file_name_replace_prefix_to + name[len(file_name_replace_prefix_from):]
        files.append(name)
        if index is None:
            labels.append(images[i]["label"])
        else:
            labels.append(images[i]["label"][index])

    labels = np.array(labels, dtype=np.float32)
    if len(labels.shape) == 1:
        labels = np.expand_dims(labels, 1)

    return files, labels


def _load_phase(source, phase, index=None, file_name_replace_prefix_from=None, file_name_replace_prefix_to=None):
    with open("{}/{}/images.json".format(source, phase), "r") as f:
        images = json.load(f)
    ids = list(images)

    files, labels = _load_data(ids, images, index=index, file_name_replace_prefix_from=file_name_replace_prefix_from, file_name_replace_prefix_to=file_name_replace_prefix_to)

    return files, labels


def get_spotcheck_dataloader(
    path, phase, batch_size, device="cpu", positive_only=False, limit=None, file_name_replace_prefix_from=None, file_name_replace_prefix_to=None, **dataloader_kwargs,
):
    files_tmp, labels_tmp = _load_phase(path, phase=phase, file_name_replace_prefix_from=file_name_replace_prefix_from, file_name_replace_prefix_to=file_name_replace_prefix_to)
    if positive_only:
        with open("{}/{}/name2ids.json".format(path, phase), "r") as f:
            name2ids = list(map(int, json.load(f)["object"]))
        files_tmp = [files_tmp[i] for i in name2ids]
        labels_tmp = labels_tmp[name2ids]
    dataset_tmp = _ImageDataset(files_tmp, labels_tmp)
    if limit is not None:
        dataset_tmp = torch.utils.data.Subset(dataset_tmp, list(range(limit)))

    def collate_fn(batch):
        batch = torch.utils.data.default_collate(batch)
        return tuple([x.to(device=device) for x in batch])

    return torch.utils.data.DataLoader(
        dataset_tmp, batch_size=batch_size, collate_fn=collate_fn, **dataloader_kwargs,
    )


def get_blindspots(json_path):
    """
    returns a list of indices in test dataset representing "true" blindspots
    """
    with open(json_path, 'r') as f:
        images = json.load(f)
    blindspot_map = defaultdict(list)
    for (i, v) in enumerate(images):
        for j in images[v]['contained']:
            blindspot_map[j].append(i)
    return list(blindspot_map.values())


def get_blindspots_df(path):
    with open(path, 'r') as f:
        images = json.load(f)
    blindspots = [[] for _ in range(len(images))]
    for (i, v) in enumerate(images):
        for j in images[v]['contained']:
            blindspots[i].append(j)
    return pd.DataFrame({'blindspot': blindspots})