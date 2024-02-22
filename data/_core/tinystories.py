from typing import Callable, List
from data._utils.binary_multitask_llm import (
    MultiTaskDatasetCollateFn,
    MultitaskDataset,
    ReadTokenMultitaskDataset,
    TokenMultitaskDatasetCollateFn,
)
from data._utils.common import (
    LimitIterableDataset,
    MixDataset,
    MixIterableDataset,
    RenameCollateFn,
    ReplicateDataset,
    ZerosIterableDataset,
    ZipCollateFn,
    ZipDataset,
    ZipIterableDataset,
    dict_batch_combiner,
)
from torch.utils.data import (
    IterableDataset,
    DataLoader,
    Dataset,
    default_collate,
    Subset,
)
from data._utils.llm import DecoderLLMCollateFn, TokenizerCollateFn
import pandas as pd
import torch
import numpy as np


class TinyStoriesDataset(IterableDataset):
    """
    train has 2717494 examples
    validation has 27629 examples.
    """

    def __init__(self, data_path):
        self.delimiter = "<|endoftext|>"
        self.data_path = data_path
        self.len = len(list(iter(self)))

    def __iter__(self):
        _text = []
        for line in open(self.data_path, "r"):
            if self.delimiter not in line:
                _text.append(line.strip())
            else:
                example = "\n".join(_text)
                if len(example) > 0:
                    yield "\n".join(_text)
                _text = []

    def __len__(self):
        return self.len
        # import pdb
        # pdb.set_trace()
        # return len(list(iter(self)))


def tinystories_tokenizer_raw():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="roneneldan/TinyStories-33M"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tinystories_tokenizer():
    from transformers import AutoTokenizer

    from .train_tinystories_tokenizer import SAVE_PATH

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=SAVE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TinyStoriesCollateFn:
    def __init__(self, device="cpu", duplicate=False):
        tokenizer = tinystories_tokenizer()
        tokenizer_kwargs = {"return_tensors": "pt", "padding": True}
        self._collate_fn = TokenizerCollateFn(
            tokenizer,
            tokenizer_kwargs,
            duplicate,
            add_labels=True,
            device=device,
        )

    def __call__(self, texts):
        return self._collate_fn(texts)


def text_len_fn(cutoff, text):
    return len(text) > cutoff


def has_str_fn(s, text):
    return s.lower() in text.lower()


def tinystories_cb_dataloader(
    data_path, concept_dataset, batch_size, max_len, num_workers=0, num=None
):
    return DataLoader(
        dataset=ZipIterableDataset(
            [
                LimitIterableDataset(
                    TinyStoriesDataset(data_path),
                    num=num,
                ),
                concept_dataset,
            ]
        ),
        batch_size=batch_size,
        collate_fn=ZipCollateFn(
            collate_fns=[
                DecoderLLMCollateFn(
                    tokenizer=tinystories_tokenizer(),
                    max_len=max_len,
                ),
                RenameCollateFn(
                    collate_fn=TokenMultitaskDatasetCollateFn(max_len=max_len),
                    rename_map={"labels": "concept_labels"},
                ),
            ],
            combiner=dict_batch_combiner,
        ),
        num_workers=num_workers,
    )


def julius_raw_data(path, drop_weird=True):
    df = pd.read_csv(path).dropna(axis=0)
    if drop_weird:
        df = df.loc[~df['poem'].astype(str).apply(lambda x: ('<|endoftext|>' in x))]
    return df


class ConceptDatasetJulius(MultitaskDataset):
    def __init__(self, path, drop_weird=True):
        self.df = julius_raw_data(path, drop_weird)
        print('ConceptDatasetJulius', self.df.shape)

    def __getitem__(self, i):
        return torch.Tensor(self.df.iloc[i, 2:].astype(int).to_numpy()).long()

    def __iter__(self):
        for i in range(len(self.df)):
            yield torch.Tensor(self.df.iloc[i, 2:].astype(int).to_numpy()).long()

    def __len__(self):
        return len(self.df)


class DatasetJulius(Dataset):
    def __init__(self, path, drop_weird=True):
        self.df = julius_raw_data(path, drop_weird)
        print('DatasetJulius', self.df.shape)

    def __getitem__(self, i):
        return self.df.iloc[i]["concept_poem"]

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        return iter(self.df["concept_poem"].values)


def cb_dataset_from_julius(path):
    """
    each example is the text
    """
    return ZipDataset(DatasetJulius(path), ConceptDatasetJulius(path))


def tinystories_cb_dataloader_with_julius_mix(
    orig_path,
    julius_path,
    num_concepts,
    max_len,
    batch_size,
    orig_len=None,
    julius_start_num=None,
    julius_end_num=None,
    julius_replicas=None,
    num_workers=0,
    concept_path=None,
    drop_weird=True,
):

    orig_dataset = ZipIterableDataset(
        [
            TinyStoriesDataset(orig_path),
            ZerosIterableDataset(num_concepts),
        ]
    )
    if orig_len is not None:
        orig_dataset = LimitIterableDataset(orig_dataset, num=orig_len)

    julius_dataset = ZipDataset(
        [
            DatasetJulius(julius_path, drop_weird=drop_weird),
            ConceptDatasetJulius(julius_path, drop_weird=drop_weird),
        ]
    )

    if julius_start_num is not None:
        if julius_end_num is None:
            julius_end_num = len(julius_dataset)
        julius_dataset = Subset(
            julius_dataset, np.arange(julius_start_num, julius_end_num)
        )
    if julius_replicas is not None:
        julius_dataset = ReplicateDataset(julius_dataset, julius_replicas)

    dataset = MixIterableDataset(orig_dataset, julius_dataset)

    collate_fn = ZipCollateFn(
        [
            DecoderLLMCollateFn(
                tokenizer=tinystories_tokenizer(),
                max_len=max_len,
            ),
            RenameCollateFn(
                collate_fn=MultiTaskDatasetCollateFn(),
                rename_map={"labels": "concept_labels"},
            ),
        ],
        combiner=dict_batch_combiner,
    )

    # if have concepts, add it.  combiner will flatten batch into single dict
    if concept_path is not None:
        concept_dataset = ReadTokenMultitaskDataset(concept_path)
        dataset = ZipIterableDataset(
            [
                dataset,
                concept_dataset,
            ]
        )

        collate_fn = ZipCollateFn(
            [
                collate_fn,
                RenameCollateFn(
                    collate_fn=TokenMultitaskDatasetCollateFn(max_len=max_len),
                    rename_map={"labels": "concept_labels"},
                ),
            ],
            combiner=dict_batch_combiner,
        )

    return DataLoader(
        dataset, collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers
    )


def tinystories_cb_dataloader_with_julius_mix_read_concepts(
    orig_path,
    julius_path,
    concept_path,
    max_len,
    batch_size,
    orig_len=None,
    julius_start_num=None,
    julius_end_num=None,
    julius_replicas=None,
    num_workers=0,
):
    # below 2 datasets don't have concepts
    orig_dataset = TinyStoriesDataset(orig_path)
    if orig_len is not None:
        orig_dataset = LimitIterableDataset(orig_dataset, num=orig_len)

    julius_dataset = DatasetJulius(julius_path)
    if julius_start_num is not None:
        if julius_end_num is None:
            julius_end_num = len(julius_dataset)
        julius_dataset = Subset(
            julius_dataset, np.arange(julius_start_num, julius_end_num)
        )
    if julius_replicas is not None:
        julius_dataset = ReplicateDataset(julius_dataset, julius_replicas)

    # mix the 2 datasets
    dataset = MixIterableDataset(orig_dataset, julius_dataset)

    # read the concept dataset
    concept_dataset = ReadTokenMultitaskDataset(concept_path)

    # zip the two kinds of datasets
    assert len(dataset) == len(concept_dataset)
    zip_dataset = ZipIterableDataset([dataset, concept_dataset])

    # collate function is same as before
    collate_fn = ZipCollateFn(
        [
            DecoderLLMCollateFn(
                tokenizer=tinystories_tokenizer(),
                max_len=max_len,
            ),
            RenameCollateFn(
                collate_fn=TokenMultitaskDatasetCollateFn(max_len=max_len),
                rename_map={"labels": "concept_labels"},
            ),
        ],
        combiner=dict_batch_combiner,
    )

    return DataLoader(
        zip_dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
