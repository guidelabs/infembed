from torch.utils.data import IterableDataset
from data._utils.llm_collate_fns import TokenizerCollateFn


class TinyStoriesDataset(IterableDataset):
    def __init__(self, data_path):
        self.delimiter = "<|endoftext|>"
        self.data_path = data_path

    def __iter__(self):
        _text = []
        for line in open(self.data_path, "r"):
            if self.delimiter not in line:
                _text.append(line.strip())
            else:
                yield "\n".join(_text)
                _text = []


def tinystories_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path='roneneldan/TinyStories-33M'
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class TinyStoriesCollateFn:
    def __init__(self, device='cpu', duplicate=False):
        tokenizer = tinystories_tokenizer()
        tokenizer_kwargs={
            'return_tensors': 'pt',
            'padding': True
        }
        self._collate_fn = TokenizerCollateFn(
            tokenizer,
            tokenizer_kwargs,
            duplicate,
            add_labels=True,
            device=device,
        )

    def __call__(self, texts):
        return self._collate_fn(texts)