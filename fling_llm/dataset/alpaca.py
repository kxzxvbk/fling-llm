import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from fling.utils.registry_utils import DATASET_REGISTRY

from fling_llm.model import export_hf_tokenizer


@DATASET_REGISTRY.register('alpaca')
class AlpacaDataset(Dataset):
    """
    Overview:
        Dataset for Alpaca. For more details, please refer to the huggingface dataset:
        https://huggingface.co/datasets/tatsu-lab/alpaca
    """

    def __init__(self, cfg: dict, train: bool):
        self.tokenizer = export_hf_tokenizer(cfg.data.tokenizer)
        self.data = load_dataset(cfg.data.data_path, split='train')
        train_len = int(0.9 * len(self.data))
        if train:
            self.data = self.data[:train_len]
        else:
            self.data = self.data[train_len:]
        self.max_len = cfg.data.max_len

    def __len__(self):
        return len(self.data['instruction'])

    def __getitem__(self, idx):
        instruction = self.data['instruction'][idx].strip()
        inputs = self.data['input'][idx].strip()
        output = self.data['output'][idx].strip()

        if len(inputs) != 0:
            model_input = f"Below is an instruction that describes a task, paired with an input that provides " \
                          f"further context. Write a response that appropriately completes the request. " \
                          f"### Instruction: {instruction} ### Input: {inputs} ### Response: "
        else:
            model_input = f"Below is an instruction that describes a task, paired with an input that provides " \
                          f"further context. Write a response that appropriately completes the request. " \
                          f"### Instruction: {instruction} ### Response: "
        encoded_input = self.tokenizer.encode(model_input, add_special_tokens=True)
        encoded_output = self.tokenizer.encode(output, add_special_tokens=False)
        total_input = (encoded_input + encoded_output + [self.tokenizer.eos_token_id])[:self.max_len]
        total_output = ([-100] * len(encoded_input) + encoded_output + [self.tokenizer.eos_token_id])[:self.max_len]

        return {'input_ids': torch.tensor(total_input), 'labels': torch.tensor(total_output)}
