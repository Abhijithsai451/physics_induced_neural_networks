import torch

def get_device():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return device