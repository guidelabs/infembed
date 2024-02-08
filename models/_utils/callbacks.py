from typing import List, Optional
from models._core.decoder_llm import GreedyDecoder
from lightning.pytorch.callbacks import Callback
import wandb


class GenericCallback(Callback):
    """
    callback that runs on end of specified hooks
    """

    def __init__(self, hook_strings: List[str]):
        self.hook_strings = hook_strings

    def on_validation_epoch_end(self, trainer, pl_module):
        if "on_validation_epoch_end" in self.hook_strings:
            self._on_epoch_end(trainer, pl_module, "val")

    def on_train_epoch_end(self, trainer, pl_module):
        if "on_train_epoch_end" in self.hook_strings:
            self._on_epoch_end(trainer, pl_module, "train")

    def on_test_epoch_end(self, trainer, pl_module):
        if "on_test_epoch_end" in self.hook_strings:
            self._on_epoch_end(trainer, pl_module, "test")

    def on_predict_epoch_end(self, trainer, pl_module):
        # batch will likely be different than other hooks, so `_on_epoch_end` may not
        # work
        if "on_predict_epoch_end" in self.hook_strings:
            self._on_epoch_end(trainer, pl_module, "predict")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "on_validation_batch_end" in self.hook_strings:
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "val")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "on_train_batch_end" in self.hook_strings:
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "train")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "on_test_batch_end" in self.hook_strings:
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "test")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # batch will likely be different than other hooks, so `_on_epoch_end` may not
        # work
        if "on_predict_batch_end" in self.hook_strings:
            self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "predict")

    def _on_epoch_end(trainer, pl_module, outputs, phase: str):
        raise NotImplementedError
    
    def _on_batch_end(trainer, pl_module, outputs, phase: str):
        raise NotImplementedError


class GreedyDecoderCallback(GenericCallback):
    """
    logs a single greedy generation at epoch ends
    """
    def __init__(self, eos_token_id: int, tokenizer, hook_strings: List[str], max_len: Optional[int] = None):
        GenericCallback.__init__(self, hook_strings)
        self.eos_token_id, self.tokenizer = eos_token_id, tokenizer
        self.max_len = max_len

    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, phase: str):
        decoder = GreedyDecoder(max_len=pl_module.max_len if self.max_len is None else self.max_len)
        text_ids = [decoder(pl_module, self.eos_token_id, example) for example in batch['input_ids']]
        texts = [self.tokenizer.decode(text_id) for text_id in text_ids]
        print(texts)
        table = wandb.Table(columns=['text'], data=[[text] for text in texts])
        # pl_module.log_dict({phase: {'greedy_text': table}})
        wandb.log({'greedy_text': table})