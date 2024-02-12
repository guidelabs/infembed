from decoder_llm import DecoderLayer, MultiAttention, FeedForward, Sublayer, Generator
import torch.nn as nn
import torch


class CBGenerator(nn.Module):
    def __init__(
        self,
        decoder_layers,
        generator,
        embedder,
        positive_concept_embedder,
        negative_concept_embedder,
        concept_generator,
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

    def generate(self, x):
        """
        `x` is the output of forward.  has shape batch size X sequence length X model dimension
        it outputs two predictions:
            1) for each token, the logit for each possible prediction
            2) for each token, for each concept, the logit for whether the concept is present
        """
        positive_concept_embeddings = self.positive_concept_embedder(
            x
        )  # batch size X sequence length X number concepts X concept embedding dim
        negative_concept_embeddings = self.negative_concept_embedder(
            x
        )  # batch size X sequence length X number concepts X concept embedding dim
        concept_logits = self.concept_generator(
            torch.cat([positive_concept_embeddings, negative_concept_embeddings], dim=1)
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
        return prediction_logits, concept_logits


class ConceptLoss(nn.Module):
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
        return (
            F.binary_cross_entropy_with_logits(concept_logits, concept_labels)
            * attention_mask
            / torch.sum(attention_mask)
        )
