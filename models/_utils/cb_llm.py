import torch.nn as nn


class CBDecoderLoss(nn.Module):
    def __init__(self, token_loss, concept_loss, tradeoff):
        super().__init__()
        self.token_loss, self.concept_loss, self.tradeoff = (
            token_loss,
            concept_loss,
            tradeoff,
        )

    def forward(
        self, prediction_logits, attention_mask, labels, concept_logits, concept_labels
    ):
        return self.token_loss(prediction_logits, attention_mask, labels) + (
            self.tradeoff
            * self.concept_loss(concept_logits, attention_mask, concept_labels)
        )