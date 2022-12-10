import torch

def test_forward():
    t = torch.tensor([0])
    example_text = "In a hole in the ground there lived a hobbit"
    output = model.forward(example_text, timestep=t)
