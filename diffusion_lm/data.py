import pdb

import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class E2EDataset(Dataset):
    def __init__(
        self, train_set="train", seq_length=64, base_model="bert-base-uncased"
    ):
        self.seq_length = seq_length
        self.tokenizer = BertTokenizer.from_pretrained(base_model)
        self.dataset = load_dataset("e2e_nlg")[train_set]["human_reference"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]
        tokens = self.tokenizer(text)["input_ids"]

        # Pad the tokens if necessary
        if len(tokens) >= self.seq_length:
            tokens = tokens[: self.seq_length]
        else:
            tokens += [self.tokenizer.pad_token_id] * (self.seq_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.int32)
