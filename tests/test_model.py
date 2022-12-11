import torch
from torch.utils.data import DataLoader
from diffusion_lm.data import E2EDataset
from diffusion_lm.model import DiffusionLM

batch_size = 2
diffusion_steps = 2000
model = DiffusionLM()
e2e_dataset = E2EDataset("train")
e2e_dataloader = DataLoader(e2e_dataset, batch_size=batch_size, shuffle=True)

def test_forward():
    batch = next(iter(e2e_dataloader))
    embeddings = model.embedding(batch)
    timesteps = torch.randint(diffusion_steps, (batch_size,))
    noised_embeddings = model.q_sample(embeddings, timesteps)
    output = model.forward(noised_embeddings, timesteps)


# def test_fit():
#     e2e_dataset = load_dataset("e2e_nlg")
#     model.fit(e2e_dataset["train"], epochs=1, batch_size=1)
