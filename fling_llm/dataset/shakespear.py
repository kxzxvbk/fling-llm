import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from fling.utils.registry_utils import DATASET_REGISTRY

from fling_llm.model import export_hf_tokenizer


@DATASET_REGISTRY.register('shakespear')
class ShakespearDataset(Dataset):
    """
    Overview:
        Dataset for shakespear. For more details, please refer to the huggingface dataset:
        https://huggingface.co/datasets/Trelis/tiny-shakespeare
    """

    def __init__(self, cfg: dict, train: bool):
        self.tokenizer = export_hf_tokenizer(cfg.data.tokenizer)
        self.data = load_dataset(cfg.data.data_path, split='train' if train else 'test')
        self.max_len = cfg.data.max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['Text'][idx]
        encoded_text = self.tokenizer.encode(text, max_length=self.max_len, truncation=True, add_special_tokens=True)
        encoded_text.append(self.tokenizer.eos_token_id)
        return {
            'input_ids': torch.tensor(encoded_text),
            'labels': torch.tensor(encoded_text)
        }
