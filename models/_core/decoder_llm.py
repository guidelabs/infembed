import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import lightning as L

# import pytorch_lightning as pl
import lightning.pytorch as pl
import copy
import torch
from data._utils.common import subsequent_mask


### define attention ###
# every sub-layer (which all use a residual connection) are chained together, and thus all their inputs and
# outputs must have the same dimension, which we will denote as `model_dim` throughout.  sub-layer refers to
# multi-attention and feed


class Attention(nn.Module):
    """
    implements a single attention head
    TODO: add dropout
    """

    def __init__(self, model_dim, key_dim, value_dim):
        super().__init__()
        self.key = nn.Linear(model_dim, key_dim)
        self.query = nn.Linear(model_dim, key_dim)
        self.value = nn.Linear(model_dim, value_dim)

    def forward(self, querys, keys, values, mask):
        """
        all inputs assumed to have dimension `model_dim`, but can be different from each other, for
        flexibility.
        `mask` is 2D, because it's the same for every example
        """
        _querys = self.query(querys)
        _keys = self.key(keys)
        _values = self.value(values)  # batch size x seq length x `value_dim`
        # compute pairwise dot-products
        weights = torch.sum(_querys[:, :, None, :] * _keys[:, None, :, :], dim=3)
        # apply mask
        # import pdb
        # pdb.set_trace()
        # weights = weights * mask[None, :, :]
        weights = weights.masked_fill(mask[None, :, :] == 0, -1e9)
        # normalize for each query
        weights = F.softmax(weights, dim=2)  # batch size X seq length x seq length
        # weight with values.
        x = torch.einsum("ijk,ikl->ijl", weights, _values)  # TODO: check
        # x = weights @ values.T
        return x  # batch size X seq len x `value_dim`


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiAttention(nn.Module):
    """
    dims refer to those of the individual heads.  should be chosen so that `value_dim * num_heads = `model_dim`
    by convention (for both dims), by convention
    """

    def __init__(self, model_dim, key_dim, value_dim, num_heads):
        super().__init__()
        self.attentions = clones(Attention(model_dim, key_dim, value_dim), num_heads)
        self.projection = nn.Linear(num_heads * value_dim, model_dim)

    def forward(self, querys, keys, values, mask):
        return self.projection(
            torch.cat(
                [
                    attention(querys, keys, values, mask)
                    for attention in self.attentions
                ],
                dim=2,
            )
        )


### define wrapper ###
# each layer is given a residual connection, and layer normalization and dropout are applied to the layer's
# output.


class LayerNorm(nn.Module):
    """
    input is of size `model_dim`.  in batch normalization, goal is for each neuron to have the same
    distribution, over the population.  in layer normalization, goal is for the distribution of the neurons
    in a single layer, within a single example, to have the same distribution, over different examples.
    """

    def __init__(self, size):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(size))
        self.offset = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        # compute mean and std for each position separately
        means = torch.mean(x, dim=2, keepdim=True)
        stds = torch.std(x, dim=2, keepdim=True)
        return (self.gain[None, None, :] * (x - means) / stds) + self.offset[
            None, None, :
        ]


class Sublayer(nn.Module):
    """
    don't pass modules for dropout and layer_norm, as they are hard-coded
    """

    def __init__(self, model_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(model_dim)

    def forward(self, x, horse):
        return self.layer_norm(x + self.dropout(horse(x)))


### define a single decoder layer ###


class FeedForward(nn.Module):
    """
    does not apply nonlinearity to output
    """

    def __init__(self, model_dim, hidden_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_2(F.relu(self.linear_1(x)))


class DecoderLayer(nn.Module):
    def __init__(
        self, attention, feedforward, attention_sublayer, feedforward_sublayer
    ):
        super().__init__()
        self.attention, self.feedforward = attention, feedforward
        self.attention_sublayer, self.feedforward_sublayer = (
            attention_sublayer,
            feedforward_sublayer,
        )

    def forward(self, x, mask):
        """
        `x` is the target's "ideal" predecessor.
        `mask` is the subsequent mask
        """
        x = self.attention_sublayer(x, lambda x: self.attention(x, x, x, mask))
        x = self.feedforward_sublayer(x, lambda x: self.feedforward(x))
        return x


### define positional encoder ###


class PositionEncoder(nn.Module):
    def __init__(self, model_dim, dropout, max_len=5000):
        """
        should multiply position by c, which ranges from 1 to 10000, geometrically.  total size of encoding
        is `model_dim`; allocate half for sin, half for cos.
        """
        super().__init__()
        # create c
        MAX_PERIOD = 10000
        c = torch.exp(
            torch.linspace(0, torch.log(torch.tensor(MAX_PERIOD)), int(model_dim / 2))
        )
        # the encoding doesn't depend on the example, so pre-compute it for a generic example.
        # precompute for example of length `max_len`, use as much as needed in `forward`
        encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len)
        encoding[:, 0::2] = torch.sin(position[:, None] * c[None, :])
        encoding[:, 1::2] = torch.cos(position[:, None] * c[None, :])
        self.register_buffer("pe", encoding)
        self._max_len = max_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[None, : x.shape[1], :])

    @property
    def max_len(self):
        return self._max_len


### define model and constructor ###


class Generator(nn.Module):
    def __init__(self, model_dim, num_tokens):
        super().__init__()
        self.projection = nn.Linear(model_dim, num_tokens)

    def forward(self, x):
        return F.log_softmax(self.projection(x), dim=2)


class Decoder(nn.Module):
    def __init__(self, decoder_layers, generator, embedder):
        super().__init__()
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.generator = generator
        self.embedder = embedder

    def forward(self, x, mask):
        x = self.embedder(x)
        for layer in self.decoder_layers:
            x = layer(x, mask)
        return x

    def full_generate(self, x, mask):
        return self.generate(self.forward(x, mask))

    def generate(self, x):
        """
        `x` is the output of forward
        """
        return self.generator(x)

    @property
    def max_len(self):
        return self.embedder.max_len


class Embedder(nn.Module):
    def __init__(self, num_tokens, model_dim, dropout, max_len):
        self.embedding = nn.Embedding(num_tokens, model_dim)
        self.position_encoder = PositionEncoder(model_dim, dropout, max_len)

    def forward(self, x):
        return self.position_encoder(self.embedding(x))

    @property
    def max_len(self):
        return self.position_encoder.max_len


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
) -> Decoder:
    decoder_layer = DecoderLayer(
        MultiAttention(model_dim, key_dim, value_dim, num_heads),
        FeedForward(model_dim, hidden_dim, dropout),
        Sublayer(model_dim, dropout),
        Sublayer(model_dim, dropout),
    )
    generator = Generator(model_dim, num_tokens)
    embedder = nn.Sequential(
        nn.Embedding(num_tokens, model_dim),
        PositionEncoder(model_dim, dropout, max_len),
    )
    return Decoder(
        clones(decoder_layer, num_layers),
        generator,
        embedder,
    )


### define how to compute loss given huggingface tokenizer output ###


class LLMCrossEntropyLoss(nn.Module):
    def forward(self, output, attention_mask, labels):
        # get per-example, per-position losses
        # losses = F.cross_entropy(output, labels, reduction='none')
        losses = F.cross_entropy(
            output.view(-1, output.shape[-1]), labels.view(-1), reduction="none"
        )
        if False:
            brute = sum([F.cross_entropy(_output, _labels, reduction='sum') for (_output, _labels) in zip(output, labels)])
            print(losses.sum(), brute)
            import pdb
            pdb.set_trace()
        # multiply by attention mask to ignore padding locations
        # losses *= attention_mask
        losses *= attention_mask.view(-1)
        # sum up losses over non-padding locations
        loss = losses.sum()
        # divide by total number of non-padding locations
        loss /= attention_mask.sum()
        return loss


class LabelSmoothingLoss(nn.Module):
    """
    accepts output of tokenizer / dataloader, applies label smoothing to labels
    inserts start token
    sets true probabilities of positions where target is padding token to 0. `attention_mask` indicates which
    are padding tokens
    """

    def __init__(self, criterion, smoothing):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.register_buffer("smoothing", torch.tensor(smoothing))
        # self.register_buffer('asdf', nn.Parameter(torch.ones(3)))
        # self.ggg = nn.Linear(3,4)

    def forward(self, output, attention_mask, labels):
        """
        `output` is the output of the generator - logits
        shifting the targets to produce input is responsibility of training loop that calls this loss
        """
        batch_size, batch_len, num_tokens = output.shape
        # create 3D version of target, also doing label smoothing
        _labels = (
            torch.ones(batch_size, batch_len, num_tokens).type_as(output)
            * self.smoothing
            / (num_tokens - 1)
        )
        # _labels.scatter_(1, labels.unsqueeze(1), 1 - self.smoothing)
        labels = labels.unsqueeze(2)
        _labels.scatter_(
            2,
            labels,
            torch.ones(labels.shape).to(dtype=_labels.dtype, device=_labels.device)
            * (1 - self.smoothing),
        )
        # set positions don't care about to zero
        _labels = _labels * attention_mask[:, :, None]
        # turn logits into log probabilities, which KLDivLoss requires
        output = F.log_softmax(output, dim=2)
        return self.criterion(output, _labels)


### define lightning module ###


class DecoderLightningModule(L.LightningModule):
    def __init__(self, decoder, loss_fn=None, configure_optimizers=None):
        super().__init__()
        self.decoder, self.loss_fn, self._configure_optimizers = (
            decoder,
            loss_fn,
            configure_optimizers,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-1
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(optimizer, T_max=8000, eta_min=6e-5)
        return [optimizer], [scheduler]

    def _step(self, batch, batch_idx):
        d = self.forward(batch)
        return {
            "loss": self.loss_fn(d["output"], batch["attention_mask"], batch["labels"])
        }

    def training_step(self, batch, batch_idx):
        d = self._step(batch, batch_idx)
        self.log_dict(
            {f"train_{key}": val for (key, val) in d.items() if key[0] != "_"},
            on_step=True,
            on_epoch=True,
        )
        return d

    def validation_step(self, batch, batch_idx):
        d = self._step(batch, batch_idx)
        self.log_dict(
            {f"validation_{key}": val for (key, val) in d.items() if key[0] != "_"},
            on_step=True,
            on_epoch=True,
        )
        return d

    def prediction_step(self, batch, batch_idx):
        import pdb
        pdb.set_trace()
        d = self._step(batch, batch_idx)
        self.log_dict(
            {f"prediction_{key}": val for (key, val) in d.items() if key[0] != "_"},
            on_step=True,
            on_epoch=True,
        )
        return d

    def forward(self, batch):
        # output is the log probabilities for each position and token
        output = self.decoder.full_generate(batch["input_ids"], batch["mask"])
        # loss = self.loss_fn(output, batch['attention_mask'], batch['labels'])
        return {
            #     'loss': loss,
            "output": output,
        }

    # @property
    def max_len(self):
        return self.decoder.max_len


class GreedyDecoder:
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, model, eos_token, input_ids):
        """
        `input_ids` is 1D, representing a single example.
        """
        output_ids = []
        for _ in range(self.max_len - len(input_ids)):
            output = next(
                iter(
                    model.decoder.full_generate(
                        x=input_ids.unsqueeze(0),
                        mask=subsequent_mask(len(input_ids)).to(
                            device=input_ids.device
                        ),
                    )
                )
            )
            output = output[-1]
            top_id = torch.argmax(output)
            if top_id == eos_token:
                break
            input_ids = torch.cat([input_ids, top_id.unsqueeze(0)])
            output_ids.append(top_id)
        return torch.Tensor(output_ids).long()
