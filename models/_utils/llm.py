from data._utils.llm import subsequent_mask
from models._utils.callbacks import GenericCallback
from models._utils.common import MLP, clones
import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd


"""
this file stores code applicable to all LLM's / sequence models
"""


class LLMCrossEntropyLoss(nn.Module):
    """
    for sequence data, computes cross entropy loss.  currently averages over all
    positions, instead of having each batch contribute equally
    """

    def forward(self, prediction_logits, attention_mask, labels):
        # get per-example, per-position losses
        # losses = F.cross_entropy(output, labels, reduction='none')
        losses = F.cross_entropy(
            prediction_logits.reshape(-1, prediction_logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        )
        if False:
            brute = sum(
                [
                    F.cross_entropy(_output, _labels, reduction="sum")
                    for (_output, _labels) in zip(output, labels)
                ]
            )
            print(losses.sum(), brute)

        # multiply by attention mask to ignore padding locations
        # losses *= attention_mask
        losses *= attention_mask.reshape(-1)
        # sum up losses over non-padding locations
        loss = losses.sum()
        # divide by total number of non-padding locations
        loss /= attention_mask.sum()
        return loss


class WriteTokensCallback(GenericCallback):
    """
    writes tokens to a csv.  assumes dataframe holding the tokens can fit in memory
    """

    def __init__(self, tokenizer, hook_strings):
        GenericCallback.__init__(self, hook_strings)
        self.tokenizer = tokenizer

    def _on_epoch_start(self, trainer, pl_module, phase: str):
        # create list to store tokens
        self.example_tokens = []

    def _on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, phase: str):
        for _input_ids, _attention_mask in zip(
            batch["input_ids"], batch["attention_mask"]
        ):
            _example_tokens = [
                self.tokenizer.decode(id)
                for (id, mask) in zip(_input_ids, _attention_mask)
                if mask == 1
            ]
            self.example_tokens.append(_example_tokens)

    def _on_epoch_end(self, trainer, pl_module, phase: str):
        # create the dataframe
        ds = []
        for (i, _example_tokens) in enumerate(self.example_tokens):
            for (t, token) in enumerate(_example_tokens):
                ds.append({'i':i, 't':t, 'token': token})
        pd.DataFrame(ds).to_csv(f"{phase}_tokens.csv")


class GreedyDecoder:
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, model, eos_token, input_ids, temperature=None):
        """
        `input_ids` is 1D, representing a single example.
        """
        import pdb
        pdb.set_trace()
        output_ids = []
        for _ in range(self.max_len - len(input_ids)):
            output = next(
                iter(
                    model.decoder.full_generate(
                        x=input_ids.unsqueeze(0),
                        mask=subsequent_mask(len(input_ids)).to(
                            device=input_ids.device
                        ),
                    )["prediction_logits"]
                )
            )

            output = output[-1]  # get logits in last layer
            if temperature is None or temperature == 0:
                top_id = torch.argmax(output)
            else:
                output = output / temperature
                top_id = Categorical(logits=output / temperature).sample()
            if top_id == eos_token:
                break
            input_ids = torch.cat([input_ids, top_id.unsqueeze(0)])
            output_ids.append(top_id)
        return torch.Tensor(output_ids).long()


def LLM_get_preds(outputs):
    return outputs["prediction_logits"].detach().cpu()