import torch


def get_device():
    # Prefer CUDA, then MPS, otherwise CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Check MPS
    elif torch.backends.mps.is_available():
        return torch.device("mps")

    # If other, choose CPU.
    else:
        return torch.device("cpu")
