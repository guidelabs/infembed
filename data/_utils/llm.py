import torch


class TokenizerCollateFn:
    """
    just applies a huggingface tokenizer to list of text. if `tracin_format` is true,
    duplicates the tokenizer output to produce length 2 tuple.  if `add_labels` is
    true, duplicates the `labels` as `input_ids`.
    """

    def __init__(
        self,
        tokenizer,
        tokenizer_kwargs,
        duplicate=False,
        add_labels=True,
        device="cpu",
    ):
        self.tokenizer, self.tokenizer_kwargs = tokenizer, tokenizer_kwargs
        self.duplicate, self.add_labels, self.device = duplicate, add_labels, device

    def __call__(self, texts):
        output = self.tokenizer(list(texts), **self.tokenizer_kwargs).to(self.device)
        if self.add_labels:
            output["labels"] = output["input_ids"]
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
        d = self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
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
                d["input_ids"][:, : max(d["input_ids"].shape[1] - 1, 0)],
                # take everything but last position of `input_ids`.  if length 0,
                # take nothing
            ],
            dim=1,
        )  # [:,:d["input_ids"].shape[1]]
        # create the mask used for generation during training. it's the same for each example, so is 2D
        mask = subsequent_mask(shifted_input_ids.shape[1])
        return {
            "labels": d["input_ids"],
            "attention_mask": d["attention_mask"],
            "input_ids": shifted_input_ids.to(dtype=int),
            "mask": mask,
        }


def LLM_get_target(batch):
    labels = batch["labels"]
    return labels.masked_fill(batch["attention_mask"] == 0, IGNORE_INDEX).cpu()

IGNORE_INDEX = -100
