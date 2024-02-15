from infembed.models._utils.common import _GenericLightningModule, clones
from infembed.models._utils.llm import LLMBinaryMultitaskMLPGenerator
from models._core.decoder_llm import (
    Decoder,
    DecoderLayer,
    FeedForward,
    MultiAttention,
    PositionEncoder,
    Sublayer,
)
import torch.nn as nn


def constructor(
    model_dim,
    key_dim,
    value_dim,
    num_heads,
    num_layers,
    dropout,
    hidden_dim,
    num_tokens,
    max_len,
    num_concepts,
    concept_generator_hidden_dims,
):
    decoder_layer = DecoderLayer(
        MultiAttention(model_dim, key_dim, value_dim, num_heads),
        FeedForward(model_dim, hidden_dim, dropout),
        Sublayer(model_dim, dropout),
        Sublayer(model_dim, dropout),
    )
    embedder = nn.Sequential(
        nn.Embedding(num_tokens, model_dim),
        PositionEncoder(model_dim, dropout, max_len),
    )

    if concept_generator_hidden_dims is None:
        concept_generator_hidden_dims = []
    concept_generator = LLMBinaryMultitaskMLPGenerator(
        model_dim,
        concept_generator_hidden_dims,
        num_concepts,
        pre_nonlinearity=True,
        post_nonlinearity=False,
    )

    return Decoder(
        clones(decoder_layer, num_layers),
        concept_generator,
        embedder,
    )


class BinaryMultitaskDecoderLightningModule(_GenericLightningModule):
    """
    lightning module for the binary multitask (LLM) scenario.  not specialized to
    decoders.  assumes that batch has keys for 'example_labels' (per-example labels)
    and 'labels' (per-token labels), though they can be none.
    will be instantiated by hydra yaml.
    """
    def _step(self, batch, batch_idx):
        d = self.forward(batch)
        return {
            "loss": self.loss_fn(
                d["prediction_logits"],
                batch["attention_mask"],
                batch["labels"],
                batch["example_labels"],
            ),
            **d,
            **batch,
        }

    def forward(self, batch):
        # output is the log probabilities for each position and token
        prediction_logits = self.model.full_generate(batch["input_ids"], batch["mask"])
        return {
            "prediction_logits": prediction_logits,
        }


class LLMMILImputeLoss(nn.Module):
    """
    for the LLM MIL scenario (have per-example and optionally, per-token labels), imputes
    the per-token labels to be the same as the per-example label, and applies a provided
    per-token loss
    """
    def __init__(self, llm_loss):
        super().__init__()
        self.llm_loss = llm_loss

    def forward(self, prediction_logits, attention_mask, labels, example_labels):
        _labels = torch.zeros(prediction_logits.shape)
        for (__labels, example_label) in zip(_labels, example_labels):
            __labels.masked_fill(example_label[None, :].bool(), 1)
        return self.llm_loss(prediction_logits, attention_mask, _labels)
    

class LLMMilMaxLoss(nn.Module):
    """
    for the LLM MIL scenario (have per-example and optionally, per-token labels), takes
    the max of the per-token predictions to get a per-example prediction, and gives it
    to a provided per-example loss.
    TODO: can try a leaky max or other aggregation function
    """
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, prediction_logits, attention_mask, labels, example_labels):
        _prediction_logits = prediction_logits.masked_fill(attention_mask.bool(), -1e9)
        _prediction_logits = torch.max(_prediction_logits, dim=1)
        return self.loss(_prediction_logits, example_labels)