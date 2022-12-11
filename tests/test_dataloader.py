import pdb
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from diffusion_lm.data import E2EDataset

def test_dataloader():
    e2e_dataset = E2EDataset("train")
    e2e_dataloader = DataLoader(e2e_dataset, batch_size=64, shuffle=True)
    for batch in e2e_dataloader:
        break
