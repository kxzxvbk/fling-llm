from datasets import load_dataset
from torch.utils.data import Dataset
from fling.utils.registry_utils import DATASET_REGISTRY

from core.model import export_hf_tokenizer


@DATASET_REGISTRY.register('shakespear')
class ShakespearDataset(Dataset):

    def __init__(self, cfg: dict, train: bool):
        self.tokenizer = export_hf_tokenizer(cfg.data.tokenizer)
        self.data = load_dataset(cfg.data.data_path, split='train' if train else 'test')
        self.max_len = cfg.data.max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['Text']
        encoded_text = self.tokenizer.encode(text, max_length=self.max_len, truncation=True, add_special_tokens=True)
        encoded_text.append(self.tokenizer.eos_token_id)
        return {
            'input_ids': encoded_text,
            'label': encoded_text
        }
