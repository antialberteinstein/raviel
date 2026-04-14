import torch
import torch.nn as nn

from attention_head.attention_head import Head
from config.config import n_embd


class MultiHeadAttention(nn.Module):
    """Run multiple attention heads in parallel and project back."""

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()

        # Parallel heads.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # Final projection after concatenation.
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, freqs_cos, freqs_sin):
        # Concatenate head outputs then project.
        out = torch.cat([h(x, freqs_cos, freqs_sin) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
