import torch
from torch.utils.data import DataLoader
from diffusion_lm.data import E2EDataset
from diffusion_lm.model import DiffusionLM

torch.autograd.set_detect_anomaly(True)

def main():

    batch_size = 64
    diffusion_steps = 2000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionLM().to(device)
    e2e_dataset = E2EDataset("train")
    e2e_dataloader = DataLoader(e2e_dataset, batch_size=batch_size, shuffle=True)

    model.fit(e2e_dataloader, epochs=10)


if __name__ == "__main__":
    main()
