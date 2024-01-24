class TokenizerCollateFn:
    """
    just applies a huggingface tokenizer to list of text. if `duplicate` is true,
    duplicates the tokenizer output to produce length 2 tuple.  if `add_labels` is
    true, duplicates the `labels` as `input_ids`.
    """
    def __init__(self, tokenizer, tokenizer_kwargs, duplicate=False, add_labels=True, device='cpu'):
        self.tokenizer, self.tokenizer_kwargs = tokenizer, tokenizer_kwargs
        self.duplicate, self.add_labels, self.device = duplicate, add_labels, device

    def __call__(self, texts):
        output = self.tokenizer(texts, **self.tokenizer_kwargs)
        if self.add_labels:
            output['labels'] = output['input_ids']
        output = {key: val.to(device=self.device) for (key, val) in output.items()}
        return output if not self.duplicate else (output, output)