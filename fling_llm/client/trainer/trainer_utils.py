import torch
import transformers
from transformers import Trainer
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset

from .sft_fedavg_trainer import SFTFedAvgTrainer

# The mapping from the names to Trainer construction functions.
name2func = {
    'sft_fedavg_trainer': SFTFedAvgTrainer,
    'default': Trainer,
}


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Save memory cost.
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def collate_fn(batch: List) -> Dict[str, torch.Tensor]:
    """
    Overview:
        Collate a list of tensors to be a single batch.
        The tensors will first be padded into the same length and then be stacked together.
    Arguments:
        - batch: A list of input tensors.
    Returns:
        - return: A batch of stacked tensors.
    """
    # Collate a batch of tensors into the same length and stack them together to be a single batch.
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x['input_ids'].shape[0], reverse=True)
    # Get each sequence and pad it
    sequences = [x['input_ids'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Don't forget to grab the labels of the *sorted* batch
    labels = [x['labels'] for x in sorted_batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return {"input_ids": sequences_padded, "labels": labels_padded}


def get_trainer(
        name: str, model: torch.nn.Module, train_dataset: Optional[Dataset], test_dataset: Optional[Dataset],
        training_args: transformers.TrainingArguments, **kwargs
) -> Trainer:
    """
    Overview:
        Construct the trainer given required inputs.
    Arguments:
        - name: The name of required trainer.
        - model: The trainable model in trainer.
        - train_dataset: The dataset used in the train function.
        - test_dataset: The dataset used in the test function.
        - training_args: The TrainingArguments of transformers.
    Returns:
        - trainer: The constructed trainer.
    """
    return name2func[name](
        model=model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        **kwargs
    )
