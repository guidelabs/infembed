from models._utils.binary_multitask_llm import _last_token_multitask_get_preds
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