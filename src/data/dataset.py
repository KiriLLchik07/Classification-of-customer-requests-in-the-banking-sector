import torch
from torch.utils.data import Dataset

from src.data.tokenizer import TextTokenizer


class BankingDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: TextTokenizer,
        max_length: int = 50,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer.encode(text)
        encoded = encoded[: self.max_length]

        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(len(encoded), dtype=torch.long),
        }
