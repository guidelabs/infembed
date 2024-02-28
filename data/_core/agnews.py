from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding
import functools
from torch.utils.data import DataLoader


"""
this contains functions needed for the agnews dataset
"""

 
def agnews_dataset(split: str = "train"):
    return load_dataset("ag_news")[split]


def agnews_tokenizer():
    return AutoTokenizer.from_pretrained("fabriceyhc/bert-base-uncased-ag_news")


def _agnews_process(tokenizer, e):
    return tokenizer(e["text"])


def agnews_dataloader(split: str, device, **dataloader_kwargs):
    dataset = agnews_dataset(split)
    tokenizer = agnews_tokenizer()
    dataset = dataset.map(functools.partial(_agnews_process, tokenizer), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    return DataLoader(dataset, collate_fn=AGNewsCollateFn(device), **dataloader_kwargs)


class AGNewsCollateFn:
    def __init__(self, device: str = 'cpu'):
        tokenizer = agnews_tokenizer()
        self._collate_fn = DataCollatorWithPadding(tokenizer)
        self.device = device

    def __call__(self, examples):
        _batch = self._collate_fn(examples)
        _batch = {key:val.to(device=self.device) for (key, val) in _batch.items()}
        return (_batch, _batch)
    

def get_target(batch):
    return batch['labels']