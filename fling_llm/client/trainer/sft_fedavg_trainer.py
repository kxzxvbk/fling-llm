from typing import Union, Tuple

from transformers import Trainer


class SFTFedAvgTrainer(Trainer):
    """
    Overview:
        This trainer is used for Supervised Fine-tuning.
    """
    def __init__(self, *args, **kwargs):
        super(SFTFedAvgTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False) -> Union[Tuple, torch.Tensor]:
        """
        Overview:
            The forward computation graph for computing loss.
        Arguments:
            - model: The model used for training.
            - inputs: A batch of inputs extracted from dataset.
            - return_outputs: Whether to return outputs.
        Return:
            - return: If ``return_outputs`` is False, the return is the computed loss. Otherwise, the return is the
                computed loss and model outputs.
        """
        # Model forward.
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )
        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss
