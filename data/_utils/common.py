from typing import Optional
from torch.utils.data import Dataset, Subset, IterableDataset
import torch
import lightning as L


class RandomSubset(Subset):
    def __init__(self, dataset: Dataset, num: int, seed=42):
        torch.manual_seed(seed)
        Subset.__init__(self, dataset, torch.randperm(len(dataset))[:num])


class LimitIterableDataset(IterableDataset):
    def __init__(self, dataset: Dataset, num: Optional[int] = None):
        self.dataset, self.num = dataset, num

    def __iter__(self):
        for i, batch in enumerate(self.dataset):
            if self.num is not None and i >= self.num:
                return
            yield batch


class GenericDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        predict_dataloader=None,
    ):
        super().__init__()
        (
            self._train_dataloader,
            self._val_dataloader,
            self._test_dataloader,
            self._predict_dataloader,
        ) = (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            predict_dataloader,
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def predict_dataloader(self):
        return self._predict_dataloader
    

def default_batch_to_x(batch):
    return batch[:-1]


def default_batch_to_target(batch):
    return batch[-1]


def keep_pre(x):
    return x[:-1]


class ComposeCollateFn:
    def __init__(self, f, g):
        self.f, self.g = f, g

    def __call__(self, x):
        return self.f(self.g(x))
    

class TokenizerCollateFn:
    """
    just applies a huggingface tokenizer to list of text. if `tracin_format` is true,
    duplicates the tokenizer output to produce length 2 tuple.  if `add_labels` is
    true, duplicates the `labels` as `input_ids`.
    """
    def __init__(self, tokenizer, tokenizer_kwargs, duplicate=False, add_labels=True, device='cpu'):
        self.tokenizer, self.tokenizer_kwargs = tokenizer, tokenizer_kwargs
        self.duplicate, self.add_labels, self.device = duplicate, add_labels, device

    def __call__(self, texts):
        output = self.tokenizer(texts, **self.tokenizer_kwargs).to(self.device)
        if self.add_labels:
            output['labels'] = output['input_ids']
        return output if not self.duplicate else (output, output)
    

def subsequent_mask(size):
    # returns 2D
    return torch.triu(torch.ones(size, size), diagonal=1) == 0


class DecoderLLMCollateFn:
    """
    everything specific to the decoder setting is handled here
    the batch should contain: input_ids, labels, attention_mask, mask
    """

    def __init__(self, tokenizer, max_len):
        tokenizer.pad_token = tokenizer.eos_token  # a HACK
        self.tokenizer, self.max_len = tokenizer, max_len

    def __call__(self, texts):
        # this is the unshifted text
        d = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        # truncate if needed
        if d["input_ids"].shape[1] > self.max_len:
            end_pos = self.max_len
        else:
            end_pos = None
        d["input_ids"] = d["input_ids"][:, :end_pos]
        d["attention_mask"] = d["attention_mask"][:, :end_pos]
        # the input to model is shifted
        shifted_input_ids = torch.cat(
            [
                torch.ones(len(texts), 1) * self.tokenizer.bos_token_id,
                d["input_ids"][:, :max(d["input_ids"].shape[1] - 1, 0)],
                # take everything but last position of `input_ids`.  if length 0,
                # take nothing
            ],
            dim=1,
        ) #[:,:d["input_ids"].shape[1]]
        # create the mask used for generation during training. it's the same for each example, so is 2D
        mask = subsequent_mask(shifted_input_ids.shape[1])
        return {
            "labels": d["input_ids"],
            "attention_mask": d["attention_mask"],
            "input_ids": shifted_input_ids.to(dtype=int),
            "mask": mask,
        }
    

class DatasetFromText(IterableDataset):
    """
    takes in raw text, splits into chunks constituting different elements of a dataset
    """

    def __init__(self, path, text_size):
        self.path, self.text_size = path, text_size

    def __iter__(self):
        t = ""
        for line in open(self.path, "r"):
            t += line
            if len(t) > self.text_size:
                yield t[: self.text_size]
                t = t[self.text_size :]


class EmptyTextDataset(IterableDataset):
    """
    yields empty text for use in prediction step, to generate text from nothing
    """
    def __init__(self, num_examples):
        self.num_examples = num_examples

    def __iter__(self):
        for _ in range(self.num_examples):
            yield ''


class IterableDatasetToDataset(Dataset):
    def __init__(self, dataset):
        self._dataset = [batch for batch in dataset]

    def __getitem__(self, i):
        return self._dataset[i]
    

def character_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    return tokenizer.train_new_from_iterator([], vocab_size=0, initial_alphabet=[])