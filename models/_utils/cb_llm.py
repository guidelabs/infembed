from typing import Dict, List
from data._utils.llm import subsequent_mask
from models._utils.binary_multitask_llm import _last_token_multitask_get_preds
import torch.nn as nn
import torch
from torch.distributions import Categorical

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
        assert prediction_logits.shape[1] == concept_labels.shape[1]
        return self.token_loss(prediction_logits, attention_mask, labels) + (
            self.tradeoff
            * self.concept_loss(concept_logits, attention_mask, concept_labels)
        )
    

def last_token_cb_llm_get_preds_and_target(outputs, batch):
    """
    copied.  TODO: use helper functions
    """
    preds = {
        f"task_{t}": task_preds.detach().cpu()
        for (t, task_preds) in enumerate(
            _last_token_multitask_get_preds(outputs['concept_logits'], batch["attention_mask"]).T
        )
    }

    target = batch["example_labels"]
    target = {f"task_{t}": target[:, t].detach().cpu() for t in range(target.shape[1])}
    # TODO: put metrics on right device so don't have to move preds and target to cpu
    return preds, target


class Strategy:
    """
    given a model and 1D `input_ids` representing a sequence, returns the imputed
    `concept_probs` for the entire `input_ids`.
    """
    def __call__(self, model, input_ids) -> torch.Tensor:
        raise NotImplementedError
    

class ConstantStrategy(Strategy):
    """
    returns a constant for tasks specified by a list.  since input is 1D, output
    represents 1D input, i.e. is of shape sequence length X number concepts
    """
    def __init__(self, concept_to_prob: List[float]):
        self.concept_to_prob = concept_to_prob

    def __call__(self, model, input_ids):
        return torch.stack(
            [torch.ones(len(input_ids)) * prob for prob in self.concept_to_prob],
            dim=0,
        ).T


class GreedyCBDecoder:
    def __init__(self, max_len, strategy):
        self.max_len, self.strategy = max_len, strategy

    def __call__(self, model, eos_token, input_ids, temperature=None):
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
                        concept_probs=self.strategy(model, input_ids).unsqueeze(0),
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