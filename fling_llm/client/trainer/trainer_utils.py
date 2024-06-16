import copy

import torch
from transformers import Trainer

from .sft_fedavg_trainer import SFTFedAvgTrainer


name2func = {
    'sft_fedavg_trainer': SFTFedAvgTrainer,
    'default': Trainer
}


def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def collate_fn(batch):
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x['input_ids'].shape[0], reverse=True)
    # Get each sequence and pad it
    sequences = [x['input_ids'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Don't forget to grab the labels of the *sorted* batch
    labels = [x['labels'] for x in sorted_batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return {"input_ids": sequences_padded, "labels": labels_padded}


def get_trainer(name, model, train_dataset, test_dataset, training_args, **kwargs):
    return name2func[name](
        model=model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        **kwargs
    )
