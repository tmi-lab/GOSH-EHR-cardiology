import torch
from torch.utils.data import Dataset


class LSTMCustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx]
        labels = self.encodings["label"][idx]

        # sequence length (non-padding tokens)
        length = (input_ids != 0).sum()

        sample = { 
            "input_ids": input_ids,
            "labels": labels,
            "lengths": torch.tensor(length, dtype=torch.long),
        }
        return sample