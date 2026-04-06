import torch
import torch.nn as nn
import torch.nn.functional as F
from config import block_size, n_embd

class Head(nn.Module):
    """Single attention head with causal (future) masking."""

    def __init__(self, head_size):
        super().__init__()

        # Linear projections for Q, K, V.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask to prevent attending to future tokens.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        # Scaled dot-product attention scores.
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # Apply causal masking.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Convert scores to probabilities.
        wei = F.softmax(wei, dim=-1)

        # Weighted sum of values.
        v = self.value(x)
        out = wei @ v

        return out
