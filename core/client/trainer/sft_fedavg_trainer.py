from transformers import Trainer


class SFTFedAvgTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SFTFedAvgTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )
        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss
