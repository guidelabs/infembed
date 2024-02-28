from models._utils.common import GenericLightningModule, clones
from models._utils.binary_multitask_llm import LLMBinaryMultitaskMLPGenerator
from models._core.decoder_llm import (
    Decoder,
    DecoderLayer,
    FeedForward,
    MultiAttention,
    PositionEncoder,
    Sublayer,
)
import torch.nn as nn


"""
this contains functions needed for the per-token decoder llm model, which makes
binary multi-task predictions for each token.  this is used to predict the
per-token concepts needed for training the cb-llm.
"""


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
    """
    returns a decoder llm.  this is given to the `BinaryMultitaskDecoderLightningModule`
    constructor
    """
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
        task_specific_dim=False,
    )

    return Decoder(
        clones(decoder_layer, num_layers),
        concept_generator,
        embedder,
    )


class BinaryMultitaskDecoderLightningModule(GenericLightningModule):
    """
    lightning module for the binary multi-task llm model, i.e. make binary multi-task
    predictions for each token.  not specialized to decoders.

    Assumptions:
    - batch contains the keys 'labels' (per-token labels, optional) and 'example_labels'
      (per-example labels).
    - output contains the key 'prediction_logits' (per-token predictions)
    """

    _STEP_DO_NOT_LOG_KEYS = [
        "prediction_logits",
        "concept_labels",
        "attention_mask",
        "example_labels",
        "labels",
        "input_ids",
        "mask",
    ]

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
        return self.decoder.full_generate(
            batch["input_ids"], batch["mask"]
        )
        # return {
        #     "prediction_logits": prediction_logits,
        # }


def last_token_get_preds(out):
    """
    used for evaluation by torchmetrics' `MultitaskWrapper`
    """
    preds = out["prediction_logits"]
    return {f"task_{t}": preds[:, -1, t] for t in range(preds.shape[2])}
