from models._core.decoder_llm import PositionEncoder, clones
from models._utils.common import _GenericLightningModule
from .decoder_llm import DecoderLayer, MultiAttention, FeedForward, Sublayer, Generator
import torch.nn as nn
import torch
import torch.nn.functional as F


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
    concept_embedding_dim,
    concept_embedder_hidden_dims,
    concept_generator_hidden_dims,
    generator_hidden_dims,
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

    if concept_embedder_hidden_dims is None:
        concept_embedder_hidden_dims = []
    _concept_embedder_dims = (
        [model_dim] + concept_embedder_hidden_dims + [concept_embedding_dim]
    )
    positive_concept_embedder = MLPConceptEmbedder(
        _concept_embedder_dims,
        num_concepts,
        pre_nonlinearity=True,
        post_nonlinearity=True,
    )
    negative_concept_embedder = MLPConceptEmbedder(
        _concept_embedder_dims,
        num_concepts,
        pre_nonlinearity=True,
        post_nonlinearity=True,
    )

    if concept_generator_hidden_dims is None:
        concept_generator_hidden_dims = []
    concept_generator = MLPConceptGenerator(
        2 * concept_embedding_dim,
        concept_generator_hidden_dims,
        num_concepts,
        pre_nonlinearity=False,
        post_nonlinearity=False,
    )

    if generator_hidden_dims is None:
        generator_hidden_dims = (
            []  # TODO: make hidden_dims uniform across implementations
        )
    generator = MLP(
        [num_concepts * concept_embedding_dim] + generator_hidden_dims + [num_tokens]
    )
    return CBDecoder(
        clones(decoder_layer, num_layers),
        generator,
        embedder,
        positive_concept_embedder,
        negative_concept_embedder,
        concept_generator,
    )


class MLP(nn.Module):
    def __init__(self, dims, pre_nonlinearity=False, post_nonlinearity=False):
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(dim_in, dim_out)
                for (dim_in, dim_out) in zip(dims[:-1], dims[1:])
            ]
        )
        self.pre_nonlinearity, self.post_nonlinearity = (
            pre_nonlinearity,
            post_nonlinearity,
        )

    def forward(self, x):
        linears = iter(self.linears)
        if self.pre_nonlinearity:
            x = F.relu(x)
        x = next(linears)(x)
        for linear in linears:
            x = F.relu(x)
            x = linear(x)
        if self.post_nonlinearity:
            x = F.relu(x)
        return x


class MLPConceptEmbedder(nn.Module):
    def __init__(self, dims, num_concepts, pre_nonlinearity, post_nonlinearity):
        super().__init__()
        self.mlps = clones(
            MLP(
                dims,
                pre_nonlinearity=pre_nonlinearity,
                post_nonlinearity=post_nonlinearity,
            ),
            num_concepts,
        )

    def forward(self, x):
        # sequence length X model dim -> sequence length X number concepts X concept embedding dim
        return torch.stack([mlp(x) for mlp in self.mlps], dim=2)


class MLPConceptGenerator(nn.Module):
    def __init__(
        self,
        concept_embedding_dim,
        hidden_dims,
        num_concepts,
        pre_nonlinearity,
        post_nonlinearity,
    ):
        super().__init__()
        self.mlps = clones(
            MLP(
                [concept_embedding_dim] + hidden_dims + [1],
                pre_nonlinearity=pre_nonlinearity,
                post_nonlinearity=post_nonlinearity,
            ),
            num_concepts,
        )

    def forward(self, x):
        # sequence length X number concepts X 2 * concept embedding dim -> sequence length X number concepts
        return torch.cat(
            [self.mlps[c](x[:, :, c, :]) for c in range(x.shape[2])], dim=2
        )


class CBDecoder(nn.Module):
    def __init__(
        self,
        decoder_layers,
        generator,  # MLP / linear / transformer, sequence length X (number concepts X concept embedding dim) -> sequence length X vocab size
        embedder,
        positive_concept_embedder,  # MLP, sequence length X model dim -> sequence length X number concepts X concept embedding dim
        negative_concept_embedder,
        concept_generator,  # MLP / linear, batch size X sequence length X number concepts X 2 * concept embedding dim -> batch size X sequence length X number concepts
    ):
        """
        `positive_concept_embedder` produces, for each concept, an embedding if the concept is present
        `negative_concept_embedder` produces, for each concept, an embedding if the concept is absent
        `concept_generator` produces, for each concept, the probability it is present
        """
        super().__init__()
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.generator = generator
        self.embedder = embedder
        self.positive_concept_embedder, self.negative_concept_embedder = (
            positive_concept_embedder,
            negative_concept_embedder,
        )
        self.concept_generator = concept_generator

    def forward(self, x, mask):
        """
        this is as in regular decoders - it outputs a representation for each token
        """
        x = self.embedder(x)
        for layer in self.decoder_layers:
            x = layer(x, mask)
        return x

    def full_generate(self, x, mask):
        """
        `x` is the output of forward.  has shape batch size X sequence length X model dimension
        it outputs two predictions:
            1) for each token, the logit for each possible prediction
            2) for each token, for each concept, the logit for whether the concept is present
        """
        x = self.forward(x, mask)
        positive_concept_embeddings = self.positive_concept_embedder(
            x
        )  # batch size X sequence length X number concepts X concept embedding dim
        negative_concept_embeddings = self.negative_concept_embedder(
            x
        )  # batch size X sequence length X number concepts X concept embedding dim
        concept_logits = self.concept_generator(
            torch.cat([positive_concept_embeddings, negative_concept_embeddings], dim=3)
        )  # batch size X sequence length X number concepts
        concept_probs = torch.sigmoid(concept_logits)
        concept_embeddings = (
            concept_probs[:, :, :, None] * positive_concept_embeddings
        ) + (
            (1.0 - concept_probs[:, :, :, None]) * negative_concept_embeddings
        )  # batch size X sequence length X number concepts X concept embedding dim
        batch_size, seq_length = concept_embeddings.shape[:2]
        concept_embeddings = concept_embeddings.reshape(
            (batch_size, seq_length, -1)
        )  # batch size X sequence length X (number concepts X concept embedding dim)
        prediction_logits = self.generator(
            concept_embeddings
        )  # batch size X sequence length X vocab size
        return {
            "prediction_logits": prediction_logits,
            "concept_logits": concept_logits,
        }


class CBDecoderLightningModule(_GenericLightningModule):
    def __init__(
        self,
        decoder: CBDecoder,
        loss_fn=None,
        configure_optimizers=None,
        scheduler_constructor=None,
    ):
        _GenericLightningModule.__init__(
            self, decoder, loss_fn, configure_optimizers, scheduler_constructor
        )

    _STEP_DO_NOT_LOG_KEYS = [
        "prediction_logits",
        "attention_mask",
        "labels",
        "input_ids",
        "mask",
        "concept_logits",
        "concept_labels",
    ]

    def _step(self, batch, batch_idx):
        d = self.forward(batch)
        return {
            "loss": self.loss_fn(
                d["prediction_logits"],
                batch["attention_mask"],
                batch["labels"],
                d["concept_logits"],
                batch["concept_labels"],
            ),
            **d,
            **batch,
        }

    def forward(self, batch):
        return self.model.full_generate(batch["input_ids"], batch["mask"])


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


class ConceptLoss(nn.Module):
    def __init__(self, equal_batch_contribution: bool = True):
        super().__init__()
        self.equal_batch_contribution = equal_batch_contribution

    def forward(self, concept_logits, attention_mask, concept_labels):
        """
        `concept_logits` shape: batch size X sequence length X number concepts
        `concept_labels` shape: batch size X sequence length X number concepts
        interpretation of `concept_labels[i,t,c]`: whether for example i, concept c
        exists in x[0:i+1], i.e. the prefix ending at and including token i.

        TODO: to get `concept_labels`, we will train a token-level multi-label classifier
        and apply to each token for each concept.  training this would be non-standard,
        because we will only assume we know whether a concept is present in a sequence, but
        not which prefixes it is present in, i.e. we don't have token-level labels.  this is
        a case of learning a classifier with ambiguous labels.  to start simple, for each sequence
        for which a concept is present, we can assume the concept is present in all prefixes,
        i.e. all token-level labels for that concept are positive.  this should actually do
        okay, because models can handle noisy labels.
        """

        def _concept_loss(_concept_logits, _concept_labels, _attention_mask):
            return (
                F.binary_cross_entropy_with_logits(
                    _concept_logits, _concept_labels, reduction="none"
                )
                * _attention_mask[:, None]
            ).sum() / (
                torch.sum(_attention_mask) * _concept_logits.shape[1]
            )  # scale by number of concepts so that loss is average per concept

        if not self.equal_batch_contribution:
            return _concept_loss(concept_logits, concept_labels, attention_mask)
        else:
            return torch.mean(
                torch.Tensor(
                    [
                        _concept_loss(_concept_logits, _concept_labels, _attention_mask)
                        for (_concept_logits, _concept_labels, _attention_mask) in zip(
                            concept_logits, concept_labels, attention_mask
                        )
                    ]
                )
            )
