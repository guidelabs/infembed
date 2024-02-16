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
        task_specific_dim=False,
    )

    return Decoder(
        clones(decoder_layer, num_layers),
        concept_generator,
        embedder,
    )


class BinaryMultitaskDecoderLightningModule(GenericLightningModule):
    """
    lightning module for the binary multitask (LLM) scenario.  not specialized to
    decoders.  assumes that batch has keys for 'example_labels' (per-example labels)
    and 'labels' (per-token labels), though they can be none.
    will be instantiated by hydra yaml.
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
        prediction_logits = self.decoder.full_generate(
            batch["input_ids"], batch["mask"]
        )
        return {
            "prediction_logits": prediction_logits,
        }


def last_token_get_preds(out):
    preds = out['prediction_logits']
    return {f"task_{t}": preds[:,-1,t] for t in range(preds.shape[2])}